"""Run tests/conversations.jsonl against the LIVE provider (app.models) and print a pass rate.

    .venv/bin/python scripts/eval_conversations.py [--only id1,id2] [-v]

Each conversation's turns go through app.agent.run_turn exactly as the API would
send them; the final state, results and size are then checked against "expect".
Uses a throwaway SQLite file, never data/state.sqlite. Costs real money on
OpenRouter (about $0.0003 per turn with qwen/qwen3.5-9b).
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import tempfile
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import agent, catalog, models, store  # noqa: E402

CLIENT = "eval-client-aaaaaaaaaaaaaaaa"


def run(conv: dict, verbose: bool) -> tuple[list[tuple[str, bool, str]], dict]:
    cid, questions, results, size, asked_text = None, 0, None, None, []
    replies: list[str] = []
    results_turn = None
    timings = []
    for n, msg in enumerate(conv["turns"], 1):
        t0 = time.perf_counter()
        text = ""
        for ev, d in agent.run_turn(cid, msg, CLIENT):
            if ev == "conversation":
                cid = d["id"]
            elif ev == "question":
                questions += 1
                asked_text.append(d["text"].lower())
            elif ev == "results":
                results = d
                results_turn = results_turn or n
            elif ev == "size":
                size = d
            elif ev == "text":
                text += d["delta"]
        timings.append(time.perf_counter() - t0)
        if text.strip():
            replies.append(text.strip())
        if verbose:
            print(f"   > {msg}\n     {text.strip()[:200]}")
    view = agent.conversation_view(cid, CLIENT)
    wants, prof = view["state"]["wants"], view["state"]["profile"]
    e = conv["expect"]
    checks: list[tuple[str, bool, str]] = []

    def check(name, ok, got=""):
        checks.append((name, bool(ok), str(got)))

    for k, v in (e.get("wants") or {}).items():
        got = wants.get(k)
        if isinstance(v, list):
            ok = isinstance(got, list) and all(any(catalog.colour_matches(g, x) or g == x for g in got) for x in v)
        else:
            ok = got == v
        check(f"wants.{k}={v}", ok, got)
    for k in e.get("wants_absent") or []:
        check(f"wants.{k} absent", k not in wants, wants.get(k))
    for k, v in (e.get("profile") or {}).items():
        got = prof.get(k)
        ok = (abs(float(got) - v) <= 0.6) if isinstance(v, (int, float)) and got is not None else got == v
        check(f"profile.{k}={v}", ok, got)

    r = e.get("results")
    if r is not None:
        items = (results or {}).get("items") or []
        check("results present (3)", len(items) == 3, len(items))
        check("at least one exact match", bool([i for i in items if i.get("exact", True)]),
              [(i["id"], i.get("exact")) for i in items])
        relaxed = set(items[0].get("relaxed", [])) if items else set()
        all_items = items
        # soft criteria are judged on the exact matches; near matches may break one and must say so
        items = [i for i in items if i.get("exact", True)] or items[:0]
        if "sleeve" in r:
            check(f"all sleeve={r['sleeve']}", items and all(i["sleeve"] == r["sleeve"] for i in items),
                  [i["sleeve"] for i in items])
        if "type" in r:  # hard filter: every item
            check(f"all type={r['type']}", all_items and all(i["type"] == r["type"] for i in all_items),
                  [i["type"] for i in all_items])
        for c in r.get("not_colours", []):  # hard filter: every item
            check(f"none {c}", all(not catalog.colour_matches(i["colour"], c) for i in all_items),
                  [i["colour"] for i in all_items])
        if r.get("colours"):
            check(f"colours {r['colours']}", items and all(any(catalog.colour_matches(i["colour"], c)
                                                               for c in r["colours"]) for i in items)
                  or "colours" in relaxed, [i["colour"] for i in items])
        if "max_price" in r:
            cap = r["max_price"] * (1.25 if "price" in relaxed else 1.0)
            check(f"all <= ${r['max_price']}{' (+25% relaxed)' if 'price' in relaxed else ''}",
                  all(i["price"] <= cap for i in items), [i["price"] for i in items])
        check("exactly one best", sum(bool(i.get("best")) for i in items) == 1)
        if any(i.get("exact") for i in items):
            check("best pick is an exact match", next(i for i in items if i.get("best")).get("exact"),
                  [(i["id"], i.get("exact")) for i in items])
        check("near matches say how they differ", all(i.get("near") for i in items if not i.get("exact")),
              [i.get("near") for i in items])
        check("reasons differ and are short", len({i["reason"] for i in items}) == len(items)
              and all(len(i["reason"]) <= 70 for i in items), [i["reason"] for i in items])
        check("ids in catalog", all(catalog.get(i["id"]) for i in items))
    if e.get("gender") and (results or {}).get("items"):
        allowed = {"men": {"men", "unisex"}, "women": {"women", "unisex"}, "kids": {"boys", "girls"}}[e["gender"]]
        check(f"all for {e['gender']}", all(i["gender"] in allowed for i in results["items"]),
              [i["gender"] for i in results["items"]])
    import re as _re
    bad = [r for r in replies if _re.search(r"\b(we|our|us)\b", r, _re.I)]
    check("first person singular (no we/our/us)", not bad, bad[:1])
    praise = [r for r in replies if _re.search(r"\b(quality|perfect|great choice|stylish|premium|elegant)\b", r, _re.I)]
    check("no unsupported praise", not praise, praise[:1])
    if "first_id" in e:
        got = [i["id"] for i in (results or {}).get("items") or []]
        check(f"first item is {e['first_id']}", got[:1] == [e["first_id"]], got)
    if "exact_count" in e:
        check(f"exact_count={e['exact_count']}", (results or {}).get("exact_count") == e["exact_count"],
              (results or {}).get("exact_count"))
    if "lead" in e:
        check("code sentence about matches", replies and replies[-1].startswith(e["lead"]), replies[-1:])
    if "results_by_turn" in e:
        check(f"results by turn {e['results_by_turn']}", results_turn and results_turn <= e["results_by_turn"],
              results_turn)
    if "max_questions" in e:
        check(f"questions <= {e['max_questions']}", questions <= e["max_questions"], questions)
    for slot in e.get("never_asked") or []:
        words = {"avoid_colours": "colour", "sleeve": "sleeve", "max_price": "spend"}[slot]
        check(f"never asked {slot}", not any(words in q for q in asked_text), asked_text)

    s = e.get("size")
    if s is not None:
        final_size = view.get("size") or size
        check("size present", final_size is not None)
        if final_size:
            if "moved" in s:
                check(f"size moved={s['moved']}", final_size["moved"] == s["moved"],
                      f"{final_size['chart_size']}->{final_size['size']}")
            for note in s.get("notes", []):
                check(f"note {note}", note in final_size["notes"], final_size["notes"])
            if "chest_estimated" in s:
                check(f"chest_estimated={s['chest_estimated']}", final_size["chest_estimated"] == s["chest_estimated"])
            if "item_is_result_index" in s:
                want = view["results"]["items"][s["item_is_result_index"]]["id"]
                check(f"size for result #{s['item_is_result_index'] + 1}", final_size["item_id"] == want,
                      final_size["item_id"])
    return checks, {"turn_seconds": timings}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    store.DB_PATH = pathlib.Path(tempfile.mkdtemp()) / "eval.sqlite"
    only = {x for x in args.only.split(",") if x}
    convs = [json.loads(line) for line in (ROOT / "tests" / "conversations.jsonl").read_text().splitlines() if line.strip()]
    convs = [c for c in convs if not only or c["id"] in only]
    print(f"provider={models.provider()} chat={models.chat_model_name()} embed={models.embed_model_name()} "
          f"search_embeddings={'on' if models.search_embed() else 'off (keyword)'}")
    total = passed = conv_ok = 0
    times: list[float] = []
    failures = []
    for c in convs:
        try:
            checks, info = run(c, args.verbose)
        except Exception as ex:  # a crash fails the whole conversation
            checks, info = [("ran without error", False, repr(ex))], {"turn_seconds": []}
        times += info["turn_seconds"]
        ok = sum(1 for _, good, _ in checks if good)
        total += len(checks)
        passed += ok
        conv_ok += ok == len(checks)
        print(f"{'PASS' if ok == len(checks) else 'FAIL'} {c['id']:<26} {ok}/{len(checks)}")
        for name, good, got in checks:
            if not good:
                failures.append((c["id"], name, got))
                print(f"     x {name}  (got: {got})")
    times.sort()
    print(f"\nchecks {passed}/{total} = {passed / max(total, 1):.0%}; conversations {conv_ok}/{len(convs)} fully passing")
    if times:
        print(f"turn latency: median {times[len(times) // 2]:.2f}s, p90 {times[int(len(times) * 0.9)]:.2f}s, "
              f"max {times[-1]:.2f}s over {len(times)} turns")
    return 0 if passed == total else 1


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    sys.exit(main())
