"""Agent loop with a fake provider: offline and fast."""
import pytest

from app import agent, catalog
from tests.fakes import HALLUCINATED_ID, fake  # noqa: F401  (fixture)

CLIENT = "client-test-aaaaaaaaaaaaaaaaaaaa"
FULL = "Men's short-sleeve shirt for a wedding, nothing white, under $60"


def turn(conv_id, message, **kw):
    events = list(agent.run_turn(conv_id, message, CLIENT, **kw))
    names = [e for e, _ in events]
    data = {}
    for e, d in events:
        if e == "conversation":
            data["conversation"] = d  # the last one carries the final title
        else:
            data.setdefault(e, d)
    text = "".join(d["delta"] for e, d in events if e == "text")
    steps = [d for e, d in events if e == "step"]
    return names, data, text, steps


def new_conv(message):
    names, data, text, steps = turn(None, message)
    return data["conversation"]["id"], names, data, text


def kinds(names):
    return [n for n in names if n != "step"]


# ------------------------------------------------------------------ clarifying questions
def test_first_question_is_who_it_is_for(fake):
    cid, names, data, _ = new_conv("A shirt for a beach wedding in June")
    assert names[0] == "conversation" and names[-1] == "done"
    assert data["conversation"]["title"] == "Beach wedding in June"
    assert "results" not in names
    assert data["question"] == {"text": "Who is it for?", "options": ["Men", "Women", "Kids"]}


def test_question_cap_gender_plus_two_details(fake):
    cid, *_ = new_conv("A shirt for a beach wedding in June")
    n2, d2, *_ = turn(cid, "Men")
    assert "sleeve" in d2["question"]["text"].lower()
    n3, d3, *_ = turn(cid, "Short sleeves")
    assert "spend" in d3["question"]["text"].lower()
    n4, d4, *_ = turn(cid, "hmm, not sure")
    assert "question" not in n4, "a fourth clarifying question was asked"
    assert "results" in n4
    msgs = agent.conversation_view(cid, CLIENT)["messages"]
    assert sum(1 for m in msgs if m.get("options")) == 3


@pytest.mark.parametrize("msg,expected", [
    ("A shirt for my husband's beach wedding", "men"),
    ("Something for her birthday", "women"),
    ("Men's office shirt", "men"),
    ("A tee for my boy", "kids"),
    ("Women's blouse for work", "women"),
])
def test_gender_implied_skips_the_question(fake, msg, expected):
    cid, names, data, _ = new_conv(msg)
    assert data["state"]["wants"]["gender"] == expected
    assert not ("question" in names and data["question"]["text"] == "Who is it for?")


def test_model_gender_guess_without_cue_is_ignored(fake):
    cid, names, data, _ = new_conv("A shirt for a gender-neutral party")  # fake extract says "men"
    assert data["question"]["text"] == "Who is it for?"
    assert "gender" not in data["state"]["wants"]


def test_gender_is_a_hard_filter_and_goes_to_profile(fake):
    cid, *_ = new_conv("A shirt for a beach wedding in June")
    turn(cid, "Women")
    turn(cid, "Long sleeves")
    names, d, *_ = turn(cid, "Under $30")  # few women's long-sleeved under $30: relaxes, never gender
    items = d["results"]["items"]
    assert items and all(i["gender"] in ("women", "unisex") for i in items)
    assert d["state"]["profile"]["gender"] == "women"
    assert {"key": "gender", "label": "Women's"} in d["results"]["filters"]


def test_kids_maps_to_boys_and_girls(fake):
    cid, *_ = new_conv("A shirt for a party")
    turn(cid, "Kids")
    turn(cid, "Either")
    names, d, *_ = turn(cid, "No limit")
    assert all(i["gender"] in ("boys", "girls") for i in d["results"]["items"])
    assert "gender" not in d["state"]["profile"]


def test_never_reasks_for_given_slots(fake):
    cid, names, data, _ = new_conv(FULL)
    assert "question" not in names and "results" in names
    items = data["results"]["items"]
    assert all(i["sleeve"] == "short" for i in items)
    assert all(not catalog.colour_matches(i["colour"], "white") for i in items)
    assert all(i["gender"] in ("men", "unisex") for i in items)


def test_option_answer_skips_extract_call(fake):
    cid, *_ = new_conv("A shirt for a beach wedding in June")
    before = fake.calls.count("extract")
    turn(cid, "Men")
    turn(cid, "Short sleeves")
    assert fake.calls.count("extract") == before


# ------------------------------------------------------------------ results grounding
def test_results_come_from_candidates_and_hallucinated_id_dropped(fake, monkeypatch):
    seen = {}
    real = agent.run_search

    def spy(wants, query):
        seen["cands"] = real(wants, query)
        return seen["cands"]

    monkeypatch.setattr(agent, "run_search", spy)
    cid, names, data, _ = new_conv(FULL)
    ids = [i["id"] for i in data["results"]["items"]]
    assert HALLUCINATED_ID not in ids
    assert len(ids) == 3 and len(set(ids)) == 3
    assert set(ids) <= {c["id"] for c in seen["cands"]}
    assert sum(i["best"] for i in data["results"]["items"]) == 1
    assert data["results"]["items"][0]["best"] is True


def test_pick_garbage_falls_back_to_ranking(fake):
    fake.pick_override = lambda cands: {"picks": [{"id": -1, "reason": ""}], "best_id": 12345}
    cid, names, data, _ = new_conv(FULL)
    items = data["results"]["items"]
    assert len(items) == 3 and items[0]["best"]
    assert all(catalog.get(i["id"]) for i in items)


def test_results_title_and_filters(fake):
    cid, names, data, _ = new_conv("Men's short-sleeve shirt for a beach wedding, nothing white, under $60")
    r = data["results"]
    assert r["title"] == "Three shirts for a beach wedding"
    keys = {f["key"] for f in r["filters"]}
    assert {"gender", "sleeve", "max_price", "avoid_colours:white", "occasion"} <= keys


# ------------------------------------------------------------------ reasons
def test_restating_reasons_are_replaced(fake):
    def restate(cands):
        return {"picks": [{"id": c, "reason": "Blue cotton shirt, short sleeve, summer."} for c in cands[:3]],
                "best_id": cands[0]}

    fake.pick_override = restate
    cid, names, data, _ = new_conv(FULL)
    reasons = [i["reason"] for i in data["results"]["items"]]
    assert len(set(reasons)) == 3, "reasons must differ"
    for i in data["results"]["items"]:
        r = i["reason"]
        assert len(r) <= 70 and r.endswith(".") and r[0].isupper()
        assert "$" not in r and "sleeve" not in r.lower()
        assert i["fabric"].split()[-1] in r.lower()  # built from the item's real fabric


def test_good_reasons_are_kept(fake):
    good = ["Easy to dress up for a ceremony.", "The most relaxed of the three.",
            "Darker, better for the evening."]
    fake.pick_override = lambda cands: {"picks": [{"id": c, "reason": r} for c, r in zip(cands, good)],
                                        "best_id": cands[0]}
    cid, names, data, _ = new_conv(FULL)
    assert sorted(i["reason"] for i in data["results"]["items"]) == sorted(good)


@pytest.mark.parametrize("reason,ok", [
    ("Blue handloom cotton, short sleeve, summer.", False),
    ("Blue shirt with short sleeves.", False),
    ("A bestseller this season.", False),
    ("Just $30 and light.", False),
    ("x" * 80, False),
    ("", False),
    ("Light cotton, plain enough for a ceremony.", True),
])
def test_clean_reason_guard(reason, ok):
    item = {"name": "Blue cotton shirt", "colour": "blue", "brand": "Fabindia"}
    assert (agent.clean_reason(reason, item) is not None) == ok


# ------------------------------------------------------------------ exact vs near (owner's black-shirt bug)
BLACK = ["I want a formal casual plain black shirt", "Men", "Either", "under 40"]


def test_exact_matches_first_and_truthful_sentence(fake):
    cid = None
    for msg in BLACK:
        names, d, text, steps = turn(cid, msg)
        cid = d["conversation"]["id"]
    r = d["results"]
    assert "usage" not in d["state"]["wants"], "formal + casual must not become a hard filter"
    exact = [i for i in r["items"] if i["exact"]]
    assert r["exact_count"] == len(exact) == 1 and exact[0]["id"] == 11371
    assert r["items"][0]["best"] and r["items"][0]["exact"], "best pick must be the exact match"
    near = [i for i in r["items"] if not i["exact"]]
    f = agent.filters_for(d["state"]["wants"])
    for i in near:
        assert i["near"] == agent.near_phrase(i, f), "near text is built from the item's real fields"
    assert {"key": "pattern", "label": "Plain"} in r["filters"], "filters stay the user's criteria"
    assert text.startswith("One shirt matches everything you asked for. I added two close ones:")
    assert "Nothing match" not in text


def test_all_exact_no_match_sentence(fake):
    names, d, text, _ = turn(None, FULL)
    assert d["results"]["exact_count"] == 3
    assert all(i["exact"] for i in d["results"]["items"])
    assert "matches everything" not in text and "close one" not in text


def test_none_exact_says_what_is_closest(fake):
    cid = None
    for msg in ["Men's plain purple shirt", "Long sleeves", "under 20"]:
        names, d, text, _ = turn(cid, msg)
        cid = d["conversation"]["id"]
    items = d["results"]["items"]
    assert items and d["results"]["exact_count"] == 0
    lc = items[0]["near"][:1].lower() + items[0]["near"][1:]
    assert text.startswith(f"Nothing matches everything you asked for. The closest is {lc}.")


OWNER = ["Men's long-sleeve plain black button shirt", "under $30"]


def test_owner_case_closest_is_the_plain_black_38(fake):
    cid = None
    for msg in OWNER:
        names, d, text, steps = turn(cid, msg)
        cid = d["conversation"]["id"]
    items = d["results"]["items"]
    assert items[0]["id"] == 11371 and items[0]["near"] == "A little over budget at $38"
    assert all(i["gender"] == "men" and i["type"] == "shirt" for i in items)
    assert text.startswith("Nothing matches everything you asked for. The closest is a little over budget at $38.")
    done = {s["id"]: s for s in steps if s["status"] == "done"}
    assert done["search"]["detail"] == "None match every filter"
    assert done["widen"]["detail"].endswith("the closest breaks only the budget")


def test_owner_case_under_40_is_the_single_exact(fake):
    cid = None
    for msg in [OWNER[0], "under $40"]:
        names, d, text, steps = turn(cid, msg)
        cid = d["conversation"]["id"]
    items = d["results"]["items"]
    assert items[0]["id"] == 11371 and items[0]["exact"] and items[0]["best"]
    assert d["results"]["exact_count"] == 1


def test_near_phrase_from_real_fields():
    item = {"pattern": "checked", "usage": "casual", "colour": "navy blue", "price": 44}
    f = {"pattern": "solid", "colours": ["black"], "max_price": 40}
    assert agent.near_phrase(item, f) == "A little over budget at $44; navy blue, not black; checked, not plain"


# ------------------------------------------------------------------ step events
def test_step_events_order_and_counts(fake):
    names, d, text, steps = turn(None, FULL)
    order = [(s["id"], s["status"]) for s in steps]
    assert order[:8] == [("understand", "start"), ("understand", "done"), ("decide", "start"), ("decide", "done"),
                         ("search", "start"), ("search", "done"), ("pick", "start"), ("pick", "done")]
    assert order[-2:] == [("write", "start"), ("write", "done")]
    # results come after the pick step and before writing; the write step ends after the first text delta
    seq = [n for n in names]
    first_text = seq.index("text")
    write_done = [i for i, (e) in enumerate(seq) if e == "step"][-1]
    assert seq.index("results") < first_text < write_done
    done = {s["id"]: s for s in steps if s["status"] == "done"}
    assert all(isinstance(s["ms"], int) and s["ms"] >= 0 for s in done.values())
    # truthful numbers: recount independently
    f = agent.filters_for(d["state"]["wants"])
    real = sum(1 for i in catalog.load() if catalog._passes(i, f))
    assert done["search"]["label"] == f"Searching {len(catalog.load())} shirts"
    assert done["search"]["detail"].startswith(f"{real} match")
    assert done["pick"]["label"] == f"Comparing the top {min(8, real)}" and done["pick"]["detail"] == "Chose 3"
    assert done["decide"]["detail"] == "Enough to search"
    assert "men's" in done["understand"]["detail"].lower() and "under $60" in done["understand"]["detail"]


def test_widen_step_when_too_few(fake):
    cid = None
    for msg in BLACK:
        names, d, text, steps = turn(cid, msg)
        cid = d["conversation"]["id"]
    done = {s["id"]: s for s in steps if s["status"] == "done"}
    assert done["search"]["detail"] == "1 matches every filter"
    f = agent.filters_for(d["state"]["wants"])
    hard = [i for i in catalog.load() if catalog._passes(i, catalog._hard(f))]
    n_near = sum(1 for i in hard if catalog.violations(i, f))
    assert done["widen"]["detail"].startswith(f"{n_near} near matches; the closest breaks")


def test_question_turn_steps_and_persistence(fake):
    names, d, text, steps = turn(None, "A shirt for a beach wedding in June")
    done = [s for s in steps if s["status"] == "done"]
    assert [s["id"] for s in done] == ["understand", "decide"]
    assert done[1]["detail"] == "Need: who it's for"
    msgs = agent.conversation_view(d["conversation"]["id"], CLIENT)["messages"]
    assert [s["id"] for s in msgs[1]["steps"]] == ["understand", "decide"]
    assert set(msgs[1]["steps"][0]) == {"id", "label", "detail", "ms"}


def test_size_step_detail(fake):
    cid, *_ = new_conv(FULL)
    names, d, text, steps = turn(cid, "178 and 76, I like a loose fit")
    done = {s["id"]: s for s in steps if s["status"] == "done"}
    sz = d["size"]
    assert done["size"]["detail"] == (f"Chest about {sz['chest_cm']:.0f} cm, {sz['chart_size']} on the chart, "
                                      f"{sz['size']} for a loose fit")
    assert "height 178 cm" in done["understand"]["detail"].lower() and "loose fit" in done["understand"]["detail"]


# ------------------------------------------------------------------ sizing
def test_reply_asks_for_height_weight_once(fake):
    cid, names, data, text = new_conv(FULL)
    assert "size" not in names
    assert "height and weight" in text
    _, _, text2, _ = turn(cid, "Show me something in blue")
    assert "height and weight" not in text2


def test_size_event_once_height_weight_known(fake):
    cid, *_ = new_conv(FULL)
    names, data, _, _ = turn(cid, "178 and 76, I like a loose fit")
    assert "size" in names
    sz = data["size"]
    best = agent.conversation_view(cid, CLIENT)["results"]["items"][0]
    assert sz["item_id"] == best["id"]
    assert sz["moved"] == 1 and "one_up_for_loose_fit" in sz["notes"]
    assert "chart_room_cm" in sz
    assert data["state"]["profile"] == {"height_cm": 178.0, "weight_kg": 76.0, "fit": "loose", "gender": "men"}
    k = kinds(names)
    assert k.index("text") < k.index("size") < k.index("state") < k.index("done")


def test_select_reference_resizes(fake):
    cid, *_ = new_conv(FULL)
    turn(cid, "178 and 76")
    names, data, _, _ = turn(cid, "what about the second one")
    second = agent.conversation_view(cid, CLIENT)["results"]["items"][1]
    assert data["size"]["item_id"] == second["id"]


# ------------------------------------------------------------------ units
@pytest.mark.parametrize("text,expected", [
    ("I'm 5'10\" and 170 lbs", {"height_cm": 177.8, "weight_kg": 77.1}),
    ("6 ft 1 in, 190 pounds", {"height_cm": 185.4, "weight_kg": 86.2}),
    ("1.82 m and 80 kg", {"height_cm": 182.0, "weight_kg": 80.0}),
    ("178 cm, 76kg", {"height_cm": 178.0, "weight_kg": 76.0}),
    ("12 stone 4", {"weight_kg": 78.0}),
    ("my chest is 40 inches", {"chest_cm": 101.6}),
])
def test_parse_units(text, expected):
    got = agent.parse_units(text)
    for k, v in expected.items():
        assert got[k] == pytest.approx(v, abs=0.1)


def test_model_raw_units_converted_in_code():
    assert agent.convert_model_units({"height_value": 5, "height_unit": "ft", "height_inches": 10,
                                      "weight_value": 170, "weight_unit": "lb"}) == \
        {"height_cm": 177.8, "weight_kg": 77.1}
    assert agent.convert_model_units({"chest_value": 40, "chest_unit": "in"}) == {"chest_cm": 101.6}


def test_feet_and_pounds_through_a_turn(fake):
    cid, *_ = new_conv(FULL)
    names, data, _, _ = turn(cid, "I'm 5 foot 10 and 170 pounds")
    p = data["state"]["profile"]
    assert p["height_cm"] == pytest.approx(177.8) and p["weight_kg"] == pytest.approx(77.1)
    assert "size" in names


# ------------------------------------------------------------------ removing filters
def test_remove_filter_field_skips_extract(fake):
    cid, *_ = new_conv(FULL)
    n_extract = fake.calls.count("extract")
    names, d2, _, _ = turn(cid, "", remove_filter="max_price")
    assert fake.calls.count("extract") == n_extract
    assert "results" in names
    assert "max_price" not in d2["state"]["wants"]
    assert "max_price" not in {f["key"] for f in d2["results"]["filters"]}


def test_remove_filter_message_from_ui(fake):
    cid, *_ = new_conv(FULL)
    n_extract = fake.calls.count("extract")
    names, d2, _, _ = turn(cid, 'Remove the "Not white" filter.')
    assert fake.calls.count("extract") == n_extract
    assert "avoid_colours" not in d2["state"]["wants"]
    assert "results" in names


def test_remove_unknown_filter_no_search(fake):
    cid, *_ = new_conv(FULL)
    names, _, text, _ = turn(cid, 'Remove the "Purple polka dots" filter.')
    assert "results" not in names and text


def test_clear_via_extract(fake):
    cid, *_ = new_conv(FULL)
    names, d, _, _ = turn(cid, "any colour is fine actually")
    assert "avoid_colours" not in d["state"]["wants"]
    assert "results" in names


def test_provider_failure_still_answers(fake, monkeypatch):
    from app import models

    def boom(*a, **k):
        raise RuntimeError("down")

    monkeypatch.setattr(models, "chat", boom)
    monkeypatch.setattr(models, "chat_stream", boom)
    names, data, text, steps = turn(None, FULL)
    assert names[-1] == "done"
    assert "results" in names and text  # code parsing + ranking + fallback reply
    assert "keyword rules" in next(s for s in steps if s["id"] == "understand" and s["status"] == "done")["detail"]


def test_reason_contradicting_the_pattern_is_rejected():
    checked = {"name": "Black checked flannel shirt", "colour": "black", "brand": "Scullers", "pattern": "checked"}
    assert agent.clean_reason("Under budget with a solid look.", checked) is None
    assert agent.clean_reason("A fine check that stays relaxed.", checked) is not None


def test_nothing_exact_still_shows_closest_never_empty_claim(fake):
    cid = None
    for msg in ["I want a formal casual plain black button shirt", "Men", "Either", "under 20"]:
        names, d, text, steps = turn(cid, msg)
        cid = d["conversation"]["id"]
    done = {s["id"]: s for s in steps if s["status"] == "done"}
    assert done["search"]["detail"] == "None match every filter"
    assert d["results"]["items"] and d["results"]["exact_count"] == 0
    assert d["results"]["title"] != "Nothing matches yet"
    assert all(i["type"] == "shirt" and i["gender"] == "men" for i in d["results"]["items"])

def test_plain_for_men_needs_no_gender_question(fake):
    names, d, text, steps = turn(None, "I want a plain black shirt for men, either sleeve, under $40")
    assert d["state"]["wants"]["gender"] == "men"
    assert "question" not in names, "it said 'for men': no 'Who is it for?'"
    assert "results" in names


@pytest.mark.parametrize("text,expected", [
    ("I want a plain black shirt for men, either sleeve, under $40", "men"),
    ("for women", "women"), ("something for HIM", "men"), ("for her birthday", "women"),
    ("a shirt for my son", "men"), ("my daughter needs one", "women"), ("kids party", "kids"),
    ("the menu", None), ("womenswear", None), ("history buff", None), ("herbal print", None),
    ("a shirt", None), ("Women's shirt", "women"), ("mens tee", "men"),
])
def test_detect_gender_words(text, expected):
    assert agent.detect_gender(text) == expected


def test_model_colours_must_be_colour_words():
    x = agent.guard_model({"colours": ["plain", "blue"], "avoid_colours": ["hot"]}, "A plain blue tee for hot weather")
    assert x["colours"] == ["blue"] and not x.get("avoid_colours")
