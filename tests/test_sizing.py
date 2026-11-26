import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.sizing import estimate_body, length_note, recommend_size

CONTRACT_KEYS = {"size", "chart_size", "moved", "chest_cm", "chest_estimated", "room_cm",
                 "garment", "shoulder_diff_cm", "ranges", "confidence", "notes"}


def make_item():
    rows = [("S", 86, 94, 100, 72, 44), ("M", 94, 102, 108, 74, 46),
            ("L", 102, 110, 116, 76, 48), ("XL", 110, 118, 124, 78, 50)]
    return {"id": 1, "gender": "men", "size_chart": {"fit": "regular", "sizes": [
        {"size": s, "body_chest": [lo, hi], "garment_chest": g, "length": ln, "shoulder": sh}
        for s, lo, hi, g, ln, sh in rows]}}


ITEM = make_item()


def rec(chest=None, fit="regular", h=178, w=76, **kw):
    p = {"height_cm": h, "weight_kg": w, "fit": fit, **kw}
    if chest is not None:
        p["chest_cm"] = chest
    return recommend_size(ITEM, p)


@pytest.mark.parametrize("h,w,g,lo,hi", [
    (178, 76, "men", 98, 102), (170, 60, "men", 88, 93),
    (185, 100, "men", 110, 116), (165, 60, "women", 88, 93)])
def test_anchors(h, w, g, lo, hi):
    assert lo <= estimate_body(h, w, g)["chest_cm"] <= hi


def test_population_means():
    assert abs(estimate_body(175.6, 85.5, "men")["chest_cm"] - 105.9) < 0.1
    assert abs(estimate_body(162.8, 67.8, "women")["chest_cm"] - 94.7) < 0.1


def test_children_smaller_than_adults():
    assert estimate_body(140, 35, "boys")["chest_cm"] < 80
    assert estimate_body(140, 35, "girls")["shoulder_cm"] < estimate_body(170, 60, "women")["shoulder_cm"]
    assert 60 <= estimate_body(120, 22 + 3, "girls")["chest_cm"] < 70


def test_monotonic_in_weight():
    order = [s["size"] for s in ITEM["size_chart"]["sizes"]]
    prev_c, prev_i = 0, -1
    for w in range(40, 140, 2):
        c = estimate_body(178, w)["chest_cm"]
        i = order.index(rec(w=w)["chart_size"])
        assert c >= prev_c and i >= prev_i
        prev_c, prev_i = c, i


@pytest.mark.parametrize("h,w", [(99, 70), (231, 70), (170, 24), (170, 251)])
def test_invalid_input(h, w):
    with pytest.raises(ValueError):
        estimate_body(h, w)
    with pytest.raises(ValueError):
        rec(h=h, w=w)


def test_invalid_other():
    with pytest.raises(ValueError):
        estimate_body(170, 60, "alien")
    with pytest.raises(ValueError):
        rec(fit="tight")
    with pytest.raises(ValueError):
        recommend_size(ITEM, {"fit": "regular"})
    with pytest.raises(ValueError):
        rec(chest=300)


def test_contract_keys_and_types():
    r = rec(chest=98)
    assert set(r) == CONTRACT_KEYS | {"length_note"}
    assert r["garment"] == {"chest": 108, "length": 74, "shoulder": 46}
    assert r["ranges"][0] == {"size": "S", "lo": 86, "hi": 94}
    assert 0 <= r["confidence"] <= 1 and all(isinstance(n, str) for n in r["notes"])


def test_regular_and_measured_override():
    r = rec(chest=98)
    assert (r["size"], r["chart_size"], r["moved"]) == ("M", "M", 0)
    assert r["chest_estimated"] is False and r["chest_cm"] == 98.0 and r["room_cm"] == 10.0
    assert r["confidence"] == 0.95 and r["notes"] == []
    e = rec(w=70)  # ~97.5, mid-range, estimated
    assert e["chest_estimated"] is True and e["confidence"] == 0.85
    assert rec(chest=120, h=178, w=76)["chart_size"] == "XL"  # override wins
    assert rec(h=178, w=76)["notes"] == ["near_size_boundary"]  # ~100.8, 1.2 from 102


def test_loose():
    r = rec(chest=98, fit="loose")
    assert (r["size"], r["chart_size"], r["moved"]) == ("L", "M", 1)
    assert "one_up_for_loose_fit" in r["notes"]
    r = rec(chest=114, fit="loose")
    assert r["size"] == "XL" and r["moved"] == 0 and "one_up_for_loose_fit" not in r["notes"]


def test_slim():
    r = rec(chest=96, fit="slim")  # lower third of M (94..102 -> <96.67)
    assert (r["size"], r["moved"]) == ("S", -1) and "one_down_for_slim_fit" in r["notes"]
    r = rec(chest=100, fit="slim")
    assert (r["size"], r["moved"]) == ("M", 0)
    r = rec(chest=87, fit="slim")  # smallest: never down
    assert r["size"] == "S" and r["moved"] == 0


def test_boundaries():
    assert rec(chest=94)["size"] == "M"
    assert rec(chest=93.9)["size"] == "S"
    assert rec(chest=118)["chart_size"] == "XL"
    r = rec(chest=95)
    assert "near_size_boundary" in r["notes"] and r["confidence"] == 0.85
    assert "near_size_boundary" not in rec(chest=98)["notes"]


def test_off_chart():
    r = rec(chest=80)
    assert r["size"] == "S" and "below_chart" in r["notes"] and r["room_cm"] == 20.0
    assert r["confidence"] == 0.75
    r = rec(chest=125, fit="loose")
    assert r["size"] == "XL" and "above_chart" in r["notes"] and r["moved"] == 0
    r = rec(chest=80, fit="slim")
    assert r["size"] == "S" and r["moved"] == 0


def test_clamped_inputs_lower_confidence():
    r = rec(h=200, w=25)  # BMI 6.3 -> clamped
    assert "inputs_clamped" in r["notes"]
    assert r["confidence"] < rec(h=200, w=80)["confidence"]


def test_shoulder_and_length():
    r = rec()
    assert abs(r["shoulder_diff_cm"] - (r["garment"]["shoulder"] - estimate_body(178, 76)["shoulder_cm"])) < 0.11
    assert length_note(ITEM, "M", 178) == "mid_hip"
    assert length_note(ITEM, "S", 190) == "short"
    assert length_note(ITEM, "XL", 160) == "long"
    assert rec()["length_note"] == "mid_hip"


def test_catalog_smoke():
    path = Path(__file__).resolve().parent.parent / "data/catalog/catalog.json"
    if not path.exists():
        pytest.skip("no catalog")
    items = [i for i in json.loads(path.read_text()) if "size_chart" in i]
    if not items:
        pytest.skip("catalog has no size_chart yet")
    for it in items:
        for fit in ("slim", "regular", "loose"):
            r = recommend_size(it, {"height_cm": 175, "weight_kg": 70, "gender": it.get("gender"), "fit": fit})
            assert r["size"] in [s["size"] for s in it["size_chart"]["sizes"]]
            assert 0 <= r["confidence"] <= 1
