"""Try-on post-processing maths (no network, no parser model): masks, compositing, colour match, crop box."""
import numpy as np
import pytest
from PIL import Image

from app import tryon_composite as tc
from app import tryon_openrouter as tor


def seg_map(h=80, w=60):
    """Synthetic person: top in the middle, a hand and a bag strap over it, pants below."""
    s = np.zeros((h, w), np.uint8)
    s[20:50, 15:45] = tc.TOP
    s[30:38, 30:36] = tc.HANDS      # hand in front of the shirt
    s[20:50, 18:20] = tc.BAG        # strap over the shirt
    s[50:80, 18:42] = tc.PANTS
    s[5:20, 22:38] = tc.FACE
    return s


def test_edit_mask_excludes_protected_and_includes_generated_top():
    so = seg_map()
    sg = so.copy()
    sg[50:55, 15:45] = tc.TOP           # generated shirt is longer: covers part of the trousers
    sg[20:50, 10:15] = tc.TOP           # and a longer sleeve over the arm/background
    region, protected = tc.edit_mask(so, sg, aligned=True, dilate_px=0)
    assert region[25, 25] and region[25, 12]          # original top + generated sleeve
    assert not region[33, 32]                        # hand stays
    assert not region[25, 19]                        # strap stays
    assert not region[52, 25]                        # trousers in the ORIGINAL are never overwritten
    assert not region[10, 30]                        # face


def test_misaligned_uses_only_original_top():
    so = seg_map()
    sg = so.copy()
    sg[20:50, 0:15] = tc.TOP
    region, _ = tc.edit_mask(so, sg, aligned=False, dilate_px=0)
    assert not region[:, :14].any()


def test_blend_outside_mask_and_protected_pixels_are_exact():
    rng = np.random.default_rng(1)
    orig = Image.fromarray(rng.integers(0, 255, (240, 180, 3), dtype=np.uint8))
    gen = Image.fromarray(rng.integers(0, 255, (120, 90, 3), dtype=np.uint8))  # half size: gets resized
    so = seg_map()
    region, protected = tc.edit_mask(so, so, dilate_px=1)
    out, m = tc.blend(orig, gen, region, protected)
    o, r = np.asarray(orig), np.asarray(out)
    assert out.size == orig.size
    assert np.array_equal(r[m == 0], o[m == 0]), "pixels outside the mask must be the original's"
    prot_full = np.asarray(Image.fromarray(protected.astype(np.uint8) * 255).resize(orig.size, Image.NEAREST)) > 127
    core = prot_full.copy()
    # interior of protected areas (1 px away from their edge) is exactly the original
    core[1:-1, 1:-1] &= prot_full[:-2, 1:-1] & prot_full[2:, 1:-1] & prot_full[1:-1, :-2] & prot_full[1:-1, 2:]
    assert np.array_equal(r[core], o[core])
    assert (m > 0.99).any(), "the shirt region does take the generated pixels"


def test_person_iou():
    a = seg_map()
    assert tc.person_iou(a, a) == 1.0
    b = np.zeros_like(a)
    b[:, :] = 0
    b[20:50, 15:45] = tc.TOP
    assert 0 < tc.person_iou(a, b) < tc.ALIGN_MIN


def test_colour_match_pulls_charcoal_to_black_only_inside_mask():
    gen = np.full((40, 40, 3), 70, np.uint8)               # charcoal shirt
    gen[:, :10] = (200, 180, 150)                          # background
    mask = np.zeros((40, 40), bool)
    mask[:, 10:] = True
    ref = np.full((500, 3), 15, np.uint8)                   # catalog: deep black
    out = tc.colour_match(gen, mask, ref, strength=0.8)
    assert out[:, 10:].mean() < 45, "charcoal moved toward black"
    assert np.array_equal(out[:, :10], gen[:, :10]), "outside the mask untouched"


def test_lab_round_trip():
    rng = np.random.default_rng(2)
    px = rng.integers(0, 255, (1000, 3), dtype=np.uint8)
    assert np.abs(tc._lab_to_srgb(tc._srgb_to_lab(px)).astype(int) - px).max() <= 1


def test_crop_box_has_supported_ratio_and_stays_inside():
    so = seg_map(200, 150)
    box, name = tc.crop_box(so, (1500, 2000), tor.RATIOS)
    x0, y0, x1, y1 = box
    assert 0 <= x0 < x1 <= 1500 and 0 <= y0 < y1 <= 2000
    assert abs((x1 - x0) / (y1 - y0) - tor.RATIOS[name]) < 0.02


def test_drift_zero_for_identical_and_high_for_changed():
    a = Image.new("RGB", (300, 400), (120, 120, 120))
    b = a.copy()
    assert tc and tor.drift(a, b) == 0.0
    c = Image.new("RGB", (300, 400), (20, 20, 20))
    assert tor.drift(a, c) > 0.3


def test_prompt_names_the_garment_and_forbids_underlayer():
    item = {"id": 13884, "name": "Black checked flannel shirt", "colour": "black", "pattern": "checked",
            "sleeve": "long", "type": "shirt"}
    phrase, rule = tor.describe_garment(item)
    assert "black checked long-sleeved button-up shirt" in phrase and "Black checked flannel shirt" in phrase
    assert "buttoned" in rule
    text = tor.notes_text(item, {"pattern_scale": "micro-check, about 1 mm", "checks_across_chest": 85,
                                 "colours": ["black", "white"], "buttons": "black", "pocket": "one patch pocket",
                                 "logo": "crest on the pocket", "collar": "point", "sleeve_style": "roll-up tabs",
                                 "hem": "curved"})
    assert "about 85 repeats" in text and "crest on the pocket" in text and "do not lighten" in text
    prompt = tor.PROMPT.format(garment=phrase, buttoned=rule, tuck="untucked")
    assert "T-shirt or vest under the shirt" in prompt and "Change nothing else" in prompt


def test_tryon_dispatch_passes_garment_and_stage(monkeypatch):
    from app import tryon

    seen = {}

    def fake(person, garment_path, out, quality="balanced", garment=None, on_stage=None):
        seen.update(garment=garment, on_stage=on_stage)
        return {}

    monkeypatch.setenv("TRYON_PROVIDER", "openrouter")
    monkeypatch.setattr(tor, "run_tryon", fake)
    cb = lambda s: None  # noqa: E731
    tryon.run_tryon("p", "g", "o", garment={"id": 1}, on_stage=cb)
    assert seen == {"garment": {"id": 1}, "on_stage": cb}


@pytest.mark.parametrize("w,h,name", [(1536, 2048, "3:4"), (1000, 1000, "1:1"), (900, 1600, "9:16")])
def test_nearest_ratio(w, h, name):
    assert tor.nearest_ratio(w, h) == name


# ------------------------------------------------------------------ DM-VTON polish (no model)
def _scene(h=400, w=300):
    """Grey shirt, black strap over it, blue trousers below, hands in front, beige wall."""
    from app import tryon_composite as tc

    img = np.zeros((h, w, 3), np.uint8)
    img[:] = (200, 190, 170)                        # wall
    seg = np.zeros((h, w), np.uint8)
    seg[100:250, 80:220] = tc.TOP
    img[100:250, 80:220] = (150, 150, 155)          # grey shirt
    seg[100:250, 100:112] = tc.TOP                  # strap the parser calls "top"
    img[100:250, 100:112] = (20, 20, 22)
    seg[250:400, 90:210] = tc.PANTS
    img[250:400, 90:210] = (40, 50, 90)
    seg[150:170, 150:175] = tc.HANDS
    img[150:170, 150:175] = (210, 160, 130)
    seg[20:90, 120:180] = tc.FACE
    return img, seg


def _polish(monkeypatch, untucked: str, **env):
    from app import tryon_dmvton as td

    monkeypatch.setenv("TRYON_UNTUCKED", untucked)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    img, seg = _scene()
    box = (60, 80, 240, 320)
    bw, bh = box[2] - box[0], box[3] - box[1]
    rgb = np.full((bh, bw, 3), 255, np.uint8)       # DM-VTON's white background
    rgb[20:220, 25:155] = (10, 10, 10)              # new black shirt, longer than the old one
    alpha = np.zeros((bh, bw), np.float32)
    alpha[20:220, 25:155] = 1.0
    painted = np.zeros((bh, bw), bool)
    out, info = td.polish_composite(Image.fromarray(img), seg, box, rgb, alpha, painted)
    return img, seg, np.asarray(out), info


def test_polish_trousers_identical_when_tucked(monkeypatch):
    from app import tryon_composite as tc

    img, seg, out, _ = _polish(monkeypatch, "0")
    pants = seg == tc.PANTS
    assert np.array_equal(out[pants], img[pants])


def test_polish_untucked_hem_covers_only_the_top_of_the_trousers(monkeypatch):
    from app import tryon_composite as tc

    img, seg, out, info = _polish(monkeypatch, "1")
    pants = seg == tc.PANTS
    changed = (out != img).any(-1) & pants
    assert changed.any(), "the hem hangs over the waistband"
    ys = np.nonzero(changed)[0]
    assert ys.max() <= 250 + info["hem_limit_px"] + 3, "never lower than the hem zone"
    assert np.array_equal(out[390:], img[390:]), "lower trousers untouched"


def test_polish_protected_and_outside_pixels_unchanged(monkeypatch):
    from app import tryon_composite as tc

    img, seg, out, info = _polish(monkeypatch, "1")
    for cls in (tc.FACE, tc.HANDS):
        m = seg == cls
        core = m.copy()
        core[1:-1, 1:-1] &= m[:-2, 1:-1] & m[2:, 1:-1] & m[1:-1, :-2] & m[1:-1, 2:]
        assert np.array_equal(out[core], img[core])
    assert np.array_equal(out[:60], img[:60]) and np.array_equal(out[:, :40], img[:, :40])


def test_polish_keeps_the_strap_when_enabled(monkeypatch):
    img, seg, out, info = _polish(monkeypatch, "1", DMVTON_STRAPS="1")
    assert info["strap_px"] > 0
    strap = np.zeros(seg.shape, bool)
    strap[110:240, 103:109]= True
    assert np.array_equal(out[strap], img[strap]), "strap pixels keep the original"


def test_polish_no_white_pasted(monkeypatch):
    img, seg, out, info = _polish(monkeypatch, "1", DMVTON_GRAIN="0")
    region = (out != img).any(-1)
    pasted = out[region]
    assert not ((pasted.min(-1) > 235)).any(), "no white or near-white pixels are pasted"


def test_strap_mask_empty_for_dark_shirt():
    from app import tryon_dmvton as td, tryon_composite as tc

    img, seg = _scene()
    img[seg == tc.TOP] = (25, 25, 28)               # black shirt: straps have no contrast
    assert not td.strap_mask(img, seg).any()


def test_hosted_edit_mask_untucked_only_with_the_flag():
    so = seg_map()
    sg = so.copy()
    sg[50:56, 15:45] = tc.TOP                    # generated hem hangs over the trousers
    tucked, _ = tc.edit_mask(so, sg, aligned=True, dilate_px=0, untucked=False)
    loose, _ = tc.edit_mask(so, sg, aligned=True, dilate_px=0, untucked=True)
    assert not tucked[52, 25] and loose[52, 25]
    assert not loose[70, 25], "lower trousers stay protected"
