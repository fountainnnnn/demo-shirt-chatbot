"""Size engine: plain code, no model, no network.

estimate_body() predicts chest circumference and shirt shoulder width from
height and weight; recommend_size() maps that onto a catalog item's size chart.
"""
from __future__ import annotations

# Linear model: value = mean + b_weight*(w - w_mean) + b_height*(h - h_mean).
# Anchored on ANSUR II (US Army 2012 survey) summary statistics:
#   men   stature ~175.6 cm, weight ~85.5 kg, chest circumference ~105.9 cm
#   women stature ~162.8 cm, weight ~67.8 kg, chest circumference ~94.7 cm
# The slopes are APPROXIMATE, hand-fitted to reflect "chest tracks weight
# strongly, height weakly" and checked against sanity anchors in the tests.
# They were not refit from the raw dataset (which is not in this repo).
# Shoulder is the garment-style shoulder seam width (wider than the
# biacromial breadth); its means/slopes are approximate too.
_MODEL = {
    "men": {
        "h_mean": 175.6, "w_mean": 85.5,
        "chest": (105.9, 0.55, 0.05),      # (mean, cm per kg, cm per cm height)
        "shoulder": (46.0, 0.10, 0.10),
    },
    "women": {
        "h_mean": 162.8, "w_mean": 67.8,
        "chest": (94.7, 0.60, 0.05),
        "shoulder": (40.0, 0.09, 0.09),
    },
}
_ADULT_OF = {"men": "men", "boys": "men", "women": "women", "girls": "women"}
_CHILD = {"boys", "girls"}

HEIGHT_RANGE = (100.0, 230.0)
WEIGHT_RANGE = (25.0, 250.0)
BMI_RANGE = (14.0, 50.0)       # outside this the weight is clamped
CHEST_RANGE = (60.0, 150.0)


def _predict(h, w, adult):
    m = _MODEL[adult]
    dh, dw = h - m["h_mean"], w - m["w_mean"]
    c0, cw, ch = m["chest"]
    s0, sw, sh = m["shoulder"]
    return c0 + cw * dw + ch * dh, s0 + sw * dw + sh * dh


def _estimate(height_cm, weight_kg, gender="men"):
    """Return (chest, shoulder, clamped)."""
    try:
        h, w = float(height_cm), float(weight_kg)
    except (TypeError, ValueError):
        raise ValueError("height and weight must be numbers")
    if not (HEIGHT_RANGE[0] <= h <= HEIGHT_RANGE[1]):
        raise ValueError(f"height_cm {h} outside {HEIGHT_RANGE}")
    if not (WEIGHT_RANGE[0] <= w <= WEIGHT_RANGE[1]):
        raise ValueError(f"weight_kg {w} outside {WEIGHT_RANGE}")
    g = (gender or "men").lower()
    if g == "unisex":
        a = _estimate(h, w, "men")
        b = _estimate(h, w, "women")
        return (a[0] + b[0]) / 2, (a[1] + b[1]) / 2, a[2] or b[2]
    if g not in _ADULT_OF:
        raise ValueError(f"unknown gender {gender!r}")

    clamped = False
    bmi = w / (h / 100) ** 2
    if bmi < BMI_RANGE[0] or bmi > BMI_RANGE[1]:
        bmi = min(max(bmi, BMI_RANGE[0]), BMI_RANGE[1])
        w = bmi * (h / 100) ** 2
        clamped = True

    adult = _ADULT_OF[g]
    if g in _CHILD:
        # Children: evaluate the adult model at the same BMI on the adult
        # reference height, then scale linearly by height ratio.
        ref_h = _MODEL[adult]["h_mean"]
        scale = min(h / ref_h, 1.0)
        chest, shoulder = _predict(ref_h, bmi * (ref_h / 100) ** 2, adult)
        chest, shoulder = chest * scale, shoulder * scale
    else:
        chest, shoulder = _predict(h, w, adult)

    c = min(max(chest, CHEST_RANGE[0]), CHEST_RANGE[1])
    if c != chest:
        clamped = True
    return c, shoulder, clamped


def estimate_body(height_cm: float, weight_kg: float, gender: str = "men") -> dict:
    chest, shoulder, _ = _estimate(height_cm, weight_kg, gender)
    return {"chest_cm": round(chest, 1), "shoulder_cm": round(shoulder, 1)}


def length_note(item: dict, size: str, height_cm: float) -> str:
    """'short' | 'mid_hip' | 'long' from garment length relative to height."""
    length = next(s["length"] for s in item["size_chart"]["sizes"] if s["size"] == size)
    ratio = length / float(height_cm)
    if ratio < 0.41:
        return "short"
    if ratio >= 0.45:
        return "long"
    return "mid_hip"


def _chart_index(sizes, chest):
    """Index of the size whose [lo, hi) contains chest (last size is closed)."""
    if chest < sizes[0]["body_chest"][0]:
        return 0, "below_chart"
    if chest > sizes[-1]["body_chest"][1]:
        return len(sizes) - 1, "above_chart"
    for i, s in enumerate(sizes):
        lo, hi = s["body_chest"]
        if lo <= chest < hi or (i == len(sizes) - 1 and chest == hi):
            return i, None
    return len(sizes) - 1, None  # unreachable with contiguous ranges


def recommend_size(item: dict, profile: dict) -> dict:
    sizes = item["size_chart"]["sizes"]
    fit = profile.get("fit", "regular")
    if fit not in ("slim", "regular", "loose"):
        raise ValueError(f"unknown fit {fit!r}")
    gender = profile.get("gender") or item.get("gender") or "men"
    height = profile.get("height_cm")
    weight = profile.get("weight_kg")

    notes, clamped, est_shoulder = [], False, None
    measured = profile.get("chest_cm")
    have_hw = height is not None and weight is not None
    if have_hw:
        e_chest, est_shoulder, clamped = _estimate(height, weight, gender)
    if measured is not None:
        chest = float(measured)
        if not (CHEST_RANGE[0] <= chest <= CHEST_RANGE[1]):
            raise ValueError(f"chest_cm {chest} outside {CHEST_RANGE}")
        estimated = False
        clamped = False  # clamping only matters to the estimate
    else:
        if not have_hw:
            raise ValueError("need chest_cm, or both height_cm and weight_kg")
        chest, estimated = e_chest, True
    chest = round(chest, 1)

    base, off = _chart_index(sizes, chest)
    idx = base
    if off:
        notes.append(off)
    if fit == "loose" and idx < len(sizes) - 1:
        idx += 1
        notes.append("one_up_for_loose_fit")
    elif fit == "slim" and idx > 0 and not off:
        lo, hi = sizes[idx]["body_chest"]
        if chest < lo + (hi - lo) / 3:
            idx -= 1
            notes.append("one_down_for_slim_fit")

    chosen = sizes[idx]
    bounds = [s["body_chest"][0] for s in sizes] + [sizes[-1]["body_chest"][1]]
    near = (not off) and min(abs(chest - b) for b in bounds) <= 1.5
    confidence = 0.85 if estimated else 0.95
    if near:
        confidence -= 0.10
        notes.append("near_size_boundary")
    if off:
        confidence -= 0.20
    if clamped:
        confidence -= 0.10
        notes.append("inputs_clamped")
    confidence = round(min(max(confidence, 0.0), 1.0), 2)

    result = {
        "size": chosen["size"],
        "chart_size": sizes[base]["size"],
        "moved": idx - base,
        "chest_cm": chest,
        "chest_estimated": estimated,
        "room_cm": round(chosen["garment_chest"] - chest, 1),
        "garment": {"chest": chosen["garment_chest"], "length": chosen["length"],
                    "shoulder": chosen["shoulder"]},
        "shoulder_diff_cm": (round(chosen["shoulder"] - est_shoulder, 1)
                             if est_shoulder is not None else None),
        "ranges": [{"size": s["size"], "lo": s["body_chest"][0], "hi": s["body_chest"][1]}
                   for s in sizes],
        "confidence": confidence,
        "notes": notes,
        "length_note": length_note(item, chosen["size"], height) if height else None,
    }
    return result
