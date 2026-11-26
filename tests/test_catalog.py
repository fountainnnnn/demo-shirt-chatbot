"""Fast catalog tests: no model, no network."""
import json
import pathlib
import re
import shutil
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]  # run without a conftest

from app import catalog  # noqa: E402

FIELDS = {"id", "name", "brand", "gender", "type", "colour", "sleeve", "pattern", "usage", "season", "fabric",
          "price", "image", "description", "size_chart", "source_name", "synthetic"}
GENDERS = {"men", "women", "boys", "girls", "unisex"}
USAGES = {"casual", "formal", "sports", "ethnic"}
SIZE_ORDER = ["XS", "S", "M", "L", "XL", "XXL"]


@pytest.fixture(scope="module")
def items():
    return catalog.load()


def test_loads_100_unique_items(items):
    assert len(items) == 100
    assert len({i["id"] for i in items}) == 100
    assert catalog.get(30802)["brand"] == "Fabindia"
    assert catalog.get(-1) is None and catalog.get("x") is None


@pytest.mark.parametrize("idx", range(100))
def test_item_schema(items, idx):
    it = items[idx]
    assert set(it) == FIELDS
    assert isinstance(it["id"], int) and it["name"] and it["brand"]
    assert it["gender"] in GENDERS
    assert it["type"] in {"shirt", "tshirt"}
    assert it["colour"] == it["colour"].lower() and it["colour"]
    assert it["sleeve"] in {"short", "long"}
    assert it["pattern"] in {"solid", "striped", "checked", "printed"}
    assert it["usage"] in USAGES
    assert it["season"] == it["season"].lower()
    assert it["fabric"] and isinstance(it["price"], int) and it["price"] > 0
    assert it["image"] == f"images/{it['id']}.jpg"
    assert it["brand"].lower() not in it["name"].lower()
    assert not re.search(r"\b(men|women|boys?|girls?|kids?)\b", it["name"], re.I)
    assert it["brand"] in it["description"] and it["description"].endswith(".")
    assert it["synthetic"] == ["price", "fabric", "size_chart"]
    assert it["source_name"]

    chart = it["size_chart"]
    assert chart["fit"] in {"slim", "regular", "relaxed"}
    sizes = chart["sizes"]
    names = [s["size"] for s in sizes]
    assert names == sorted(names, key=SIZE_ORDER.index) and {"S", "M", "L", "XL"} <= set(names)
    for a, b in zip(sizes, sizes[1:]):
        assert a["body_chest"][1] == b["body_chest"][0], "body_chest ranges must be contiguous"
    for s in sizes:
        lo, hi = s["body_chest"]
        assert lo < hi
        assert s["garment_chest"] > (lo + hi) / 2, "garment must have positive ease"
        assert s["length"] > 0 and s["shoulder"] > 0


def test_regular_men_template_matches_contract(items):
    it = next(i for i in items if i["gender"] == "men" and i["size_chart"]["fit"] == "regular")
    assert [s["body_chest"] for s in it["size_chart"]["sizes"]] == [[86, 94], [94, 102], [102, 110], [110, 118]]


def test_kids_and_women_have_scaled_charts(items):
    def m(g):
        it = next(i for i in items if i["gender"] == g)
        return next(s for s in it["size_chart"]["sizes"] if s["size"] == "M")["body_chest"]
    assert m("women")[1] < 94 + 1 and m("boys")[1] < m("women")[1]
    assert any(i["gender"] == "boys" for i in items)


def test_enrichment_is_deterministic():
    import enrich_catalog as e
    raw = json.loads(e.RAW.read_text())
    preds = json.loads(e.CLIP_CACHE.read_text())
    ov = json.loads(e.OVERRIDES.read_text())
    assert e.build(raw, preds, ov) == catalog.load()


def test_brand_parsing():
    import enrich_catalog as e
    assert e.parse_brand("United Colors Of Benetton Men Summer Black Shirts") == ("United Colors of Benetton", "Men Summer Black Shirts")
    assert e.parse_brand("U.S. Polo Assn. Men Printed Steeple Grey Tshirt")[0] == "U.S. Polo Assn."
    assert e.parse_brand("Reid & Taylor Men Solid Cream Shirts")[0] == "Reid & Taylor"
    assert e.parse_brand("Scullers For Her Women Printed Navy Blue Shirt")[0] == "Scullers"
    assert e.parse_brand("Gini and Jony Boys Check Blue Shirt")[0] == "Gini and Jony"


# ----------------------------------------------------------------- filters
def test_sleeve_filter():
    res = catalog.search({"sleeve": "short"}, k=100)
    assert res and all(r["sleeve"] == "short" for r in res)
    res = catalog.search({"sleeve": "long", "type": "shirt"}, k=100)
    assert res and all(r["sleeve"] == "long" and r["type"] == "shirt" for r in res)


def test_avoid_white_excludes_white_off_white_cream():
    all_items = catalog.load()
    assert any(i["colour"] in ("white", "cream") for i in all_items)
    res = catalog.search({"avoid_colours": ["white"]}, k=100)
    assert res and not [r for r in res if r["colour"] in ("white", "off white", "cream")]
    assert len(res) == sum(1 for i in all_items if i["colour"] not in ("white", "off white", "cream"))


def test_blue_matches_navy_and_teal_but_not_red():
    res = catalog.search({"colours": ["blue"]}, k=100)
    cols = {r["colour"] for r in res}
    assert {"blue", "navy blue", "teal"} <= cols and "red" not in cols
    assert {r["colour"] for r in catalog.search({"colours": ["navy"]}, k=100)} == {"navy blue"}
    assert {r["colour"] for r in catalog.search({"colours": ["gray"]}, k=100)} == {"grey"}


def test_max_and_min_price():
    res = catalog.search({"max_price": 30, "min_price": 20, "type": "tshirt"}, k=100)
    assert res and all(20 <= r["price"] <= 30 for r in res)


def test_gender_and_type_filters():
    assert all(r["gender"] == "women" for r in catalog.search({"gender": "women"}, k=100))
    assert all(r["type"] == "tshirt" for r in catalog.search({"type": "t-shirt"}, k=100))
    assert not any(r["gender"] == "boys" for r in catalog.search({"gender": "men"}, k=100))


def test_results_have_score_and_respect_k():
    res = catalog.search({}, "blue", k=5)
    assert len(res) == 5 and all("score" in r and "relaxed" in r for r in res)


# -------------------------------------------------------------- near matches
def test_no_near_matches_when_enough_exact():
    res = catalog.search({"type": "shirt", "sleeve": "long"}, k=10)
    assert len(res) >= 3 and all(r["relaxed"] == [] and r["exact"] for r in res)


PLAIN_BLACK_MEN = {"gender": ["men"], "type": "shirt", "sleeve": "long", "pattern": "solid", "colours": ["black"]}


def test_owner_case_under_30_closest_is_plain_black_38():
    res = catalog.search({**PLAIN_BLACK_MEN, "max_price": 30}, k=3)
    assert not any(r["exact"] for r in res)
    assert res[0]["id"] == 11371 and res[0]["violations"] == [{"key": "price", "severity": 1}]
    assert [len(r["violations"]) for r in res] == sorted(len(r["violations"]) for r in res)
    assert all(r["colour"] != "navy blue" for r in res), "two violations never beat one"


def test_owner_case_under_40_single_exact_first():
    res = catalog.search({**PLAIN_BLACK_MEN, "max_price": 40}, k=3)
    assert res[0]["id"] == 11371 and res[0]["exact"]
    assert sum(r["exact"] for r in res) == 1
    assert all(len(r["violations"]) == 1 for r in res[1:])


def test_fewer_violations_always_rank_first():
    res = catalog.search({"type": "shirt", "pattern": "printed", "usage": "ethnic", "colours": ["pink"],
                          "max_price": 45, "avoid_colours": ["white"], "sleeve": "long"}, k=100)
    assert res
    counts = [len(r["violations"]) for r in res]
    assert counts == sorted(counts)
    for r in res:
        assert r["type"] == "shirt" and r["colour"] not in ("white", "off white", "cream"), "hard filters stay"


def test_price_over_60_percent_is_excluded_and_35_percent_is_small():
    res = catalog.search({"type": "tshirt", "max_price": 15}, k=100)
    assert all(r["price"] <= 15 * 1.6 for r in res)
    for r in res:
        for v in r["violations"]:
            if v["key"] == "price":
                assert v["severity"] == (1 if r["price"] <= 15 * 1.35 else 2)


def test_colour_near_family_is_small_navy_is_not_near_black():
    grey = next(i for i in catalog.load() if i["colour"] == "grey")
    navy = next(i for i in catalog.load() if i["colour"] == "navy blue")
    assert catalog.violations(grey, {"colours": ["black"]}) == [{"key": "colours", "severity": 1}]
    assert catalog.violations(navy, {"colours": ["black"]}) == [{"key": "colours", "severity": 2}]


def test_gender_and_type_never_relaxed():
    res = catalog.search({"gender": ["women"], "type": "tshirt", "colours": ["purple"], "max_price": 15}, k=20)
    assert all(r["gender"] in ("women", "unisex") and r["type"] == "tshirt" for r in res)


# ----------------------------------------------------------------- ranking
def test_keyword_ranking_prefers_matching_description():
    top = catalog.search({"type": "tshirt"}, "striped navy blue polo", k=3)
    assert top[0]["pattern"] == "striped" and top[0]["colour"] == "navy blue" and "polo" in top[0]["description"].lower()
    assert top[0]["score"] >= top[1]["score"] >= top[2]["score"]
    top = catalog.search({"type": "shirt"}, "long sleeve checked green", k=3)
    assert all(r["sleeve"] == "long" and r["pattern"] == "checked" and r["colour"] == "green" for r in top)


def test_empty_query_still_returns_results():
    res = catalog.search({"type": "shirt"}, "", k=4)
    assert len(res) == 4 and all(r["score"] == 0.0 for r in res)


# --------------------------------------------------- embeddings (fake model)
def fake_embed(texts):
    """Deterministic bag-of-words hashing embedding, L2-normalised."""
    out = np.zeros((len(texts), 64), dtype="float32")
    for r, t in enumerate(texts):
        for w in re.findall(r"[a-z]+", t.lower()):
            out[r, sum(map(ord, w)) % 64] += 1.0
    return out / np.linalg.norm(out, axis=1, keepdims=True)


@pytest.fixture
def emb_dir(tmp_path, monkeypatch):
    shutil.copy(catalog.CATALOG_DIR / "catalog.json", tmp_path / "catalog.json")
    monkeypatch.setattr(catalog, "CATALOG_DIR", tmp_path)
    items = catalog.load()
    np.save(tmp_path / "embeddings.npy", fake_embed([i["description"] for i in items]))
    (tmp_path / "embeddings_ids.json").write_text(json.dumps([i["id"] for i in items]))
    return tmp_path


def test_embedding_path_ranks_by_cosine(emb_dir):
    calls = []

    def embed(texts):
        calls.append(texts)
        return fake_embed(texts)

    target = catalog.get(31543)
    res = catalog.search({}, target["description"], k=3, embed=embed)
    assert calls == [[target["description"]]]
    assert res[0]["id"] == 31543 and res[0]["score"] == pytest.approx(1.0, abs=1e-4)
    assert res[0]["score"] >= res[1]["score"] >= res[2]["score"]


def test_embedding_falls_back_to_keywords_without_file(tmp_path, monkeypatch):
    shutil.copy(catalog.CATALOG_DIR / "catalog.json", tmp_path / "catalog.json")
    monkeypatch.setattr(catalog, "CATALOG_DIR", tmp_path)

    def boom(texts):
        raise AssertionError("embed must not be called without embeddings.npy")

    res = catalog.search({"type": "shirt"}, "checked green", k=3, embed=boom)
    assert res and res[0]["pattern"] == "checked"


def test_embedding_with_filters_and_near_matches(emb_dir):
    res = catalog.search({"type": "shirt", "pattern": "printed", "sleeve": "long"}, "relaxed fit cotton", k=5,
                         embed=fake_embed)
    assert len(res) == 5
    exact = [r for r in res if r["exact"]]
    assert res[: len(exact)] == exact, "exact matches first"
    groups = {}
    for r in res:
        key = (len(r["violations"]), sum(v["severity"] for v in r["violations"]))
        groups.setdefault(key, []).append(r["score"])
    assert list(groups) == sorted(groups), "fewer / milder violations first"
    for scores in groups.values():
        assert scores == sorted(scores, reverse=True), "then by cosine score"
