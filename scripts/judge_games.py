"""LLM-as-a-judge over the big+reverse batches.

For every ``exp_big_*.json`` and ``exp_big_rev_*.json`` game we read
the injected impostor strategy, the discussion utterances, and the
outcome, then ask a judge model to score three dimensions on a 1-5
scale:

* ``strategy_adherence`` — did the impostor actually follow the
  strategy that was injected into their prompts?
* ``deception_quality`` — how convincing were the impostor's
  discussion contributions (specificity, consistency, deflection)?
* ``crew_defense`` — how sharp were the crewmates' accusations and
  deductions (were they citing seen_actions and rounds, or vague)?

Judge output lands in ``data/exp_judge_scores.csv`` incrementally so
a crash mid-run doesn't lose everything. Re-running the script skips
games already scored.

Judge model: ``uni/qwen3-next-80b-a3b-instruct`` (larger than either
player model — separates judge quality from player quality).
"""

import csv
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from langchain.schema import HumanMessage, SystemMessage

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.abspath(os.path.join(HERE, "..", "src"))
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from among_them.batch_runs import BATCH_PREFIX_TAGS, strategy_name  # noqa: E402
from among_them.config import RUNS_DIR  # noqa: E402
from among_them.llm_factory import make_llm, safe_invoke  # noqa: E402
from among_them.rag.seeder import CURATED_STRATEGIES  # noqa: E402

_STRATEGY_TEXT = {s["name"]: s["text"] for s in CURATED_STRATEGIES}

JUDGE_MODEL = "uni/qwen3-next-80b-a3b-instruct"
OUT_CSV = "data/exp_judge_scores.csv"
PARALLELISM = 4

SYSTEM_PROMPT = """You are an evaluator scoring a text-based social deduction game modelled after Among Us. You are given:
- the impostor's injected strategy prompt
- a list of impostor names
- the discussion utterances from every meeting in the game
- the final outcome

Score three dimensions on an integer 1-5 scale (1 = poor, 5 = excellent).

strategy_adherence: how closely did the impostor's discussion behaviour match the injected strategy?
deception_quality: how convincing were the impostor's discussion contributions (specificity, consistency, effective deflection)?
crew_defense: how sharp were the crewmates' accusations and deductions (concrete evidence, use of round numbers and seen locations, avoiding vague suspicion)?

Reply with ONLY a JSON object on a single line, no prose or code fences:
{"strategy_adherence": <1-5>, "deception_quality": <1-5>, "crew_defense": <1-5>, "notes": "<one short sentence>"}
"""


def load_scored() -> set[str]:
    if not os.path.exists(OUT_CSV):
        return set()
    with open(OUT_CSV) as f:
        r = csv.DictReader(f)
        return {row["file"] for row in r}


def extract_inputs(game_path: str) -> dict | None:
    with open(game_path) as f:
        s = json.load(f)
    players = s["players"]
    impostor_names = [p["name"] for p in players if p.get("is_impostor")]
    if not impostor_names:
        return None

    strat_name = strategy_name(os.path.basename(game_path))
    strategy = _STRATEGY_TEXT.get(strat_name, "")

    utterances = []
    for line in s.get("playthrough", []):
        m = re.search(r"\[([^\]]+)\]:\s*(.+)$", line)
        if m and "chat" in line:
            speaker, body = m.group(1), m.group(2).strip()
            utterances.append(f"[{speaker}]: {body}")

    last = s.get("playthrough", ["?"])[-1]
    if "Impostors win" in last:
        outcome = "Impostors won"
    elif "Crewmates win" in last:
        outcome = "Crewmates won"
    else:
        outcome = "Round cap / unfinished"

    return {
        "strategy": strategy[:1200] if strategy else "(none injected)",
        "impostor_names": impostor_names,
        "utterances": utterances,
        "outcome": outcome,
    }


def build_user_prompt(inputs: dict) -> str:
    ut = "\n".join(inputs["utterances"]) or "(no discussion utterances)"
    if len(ut) > 8000:
        ut = ut[:8000] + "\n... [truncated]"
    return (
        f"Impostor(s): {', '.join(inputs['impostor_names'])}\n"
        f"Outcome: {inputs['outcome']}\n\n"
        f"Injected strategy:\n{inputs['strategy']}\n\n"
        f"Discussion utterances:\n{ut}\n"
    )


_JSON_LINE = re.compile(r"\{[^{}]*\}")


def parse_scores(raw: str) -> dict | None:
    for m in _JSON_LINE.finditer(raw):
        try:
            data = json.loads(m.group(0))
        except json.JSONDecodeError:
            continue
        if not all(
            k in data for k in ("strategy_adherence", "deception_quality", "crew_defense")
        ):
            continue
        try:
            return {
                "strategy_adherence": int(data["strategy_adherence"]),
                "deception_quality": int(data["deception_quality"]),
                "crew_defense": int(data["crew_defense"]),
                "notes": str(data.get("notes", ""))[:300],
            }
        except (TypeError, ValueError):
            continue
    return None


def judge_one(judge_llm, file_name: str) -> dict | None:
    path = os.path.join(RUNS_DIR, file_name)
    try:
        inputs = extract_inputs(path)
    except Exception as e:
        return {"file": file_name, "error": f"parse: {e}"}
    if inputs is None:
        return {"file": file_name, "error": "no impostor"}

    prompt = build_user_prompt(inputs)
    msgs = [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=prompt)]
    reply = safe_invoke(judge_llm, msgs, attempts=2, label=f"judge/{file_name}")
    if reply is None:
        return {"file": file_name, "error": "llm unavailable"}
    scores = parse_scores(reply.content)
    if scores is None:
        return {"file": file_name, "error": f"unparseable: {reply.content[:80]}"}
    return {"file": file_name, **scores, "outcome": inputs["outcome"]}


def _error_bucket(err: str) -> str:
    """Collapse per-row error strings into a small set of categories."""
    if err == "llm unavailable":
        return "llm unavailable"
    if err.startswith("parse:"):
        return "parse"
    if err.startswith("unparseable:"):
        return "unparseable"
    return err


def main() -> None:
    project_root = os.path.abspath(os.path.join(HERE, ".."))
    os.chdir(project_root)

    all_files = sorted(
        f for f in os.listdir(RUNS_DIR)
        if any(f.startswith(p) for p in BATCH_PREFIX_TAGS) and f.endswith(".json")
    )
    scored = load_scored()
    todo = [f for f in all_files if f not in scored]
    print(f"Judge model: {JUDGE_MODEL}")
    print(f"Total eligible files: {len(all_files)}")
    print(f"Already scored: {len(scored)}. To score: {len(todo)}\n", flush=True)
    if not todo:
        print("Nothing to do.")
        return

    judge_llm = make_llm(JUDGE_MODEL, temperature=0)

    # Canary: judge one game synchronously before spinning up the pool, so a
    # systemic failure (bad/expired API key, backend down, wrong model id)
    # aborts immediately instead of silently writing "llm unavailable" into
    # every row of a multi-hour batch.
    print(f"Canary check against {todo[0]}...", flush=True)
    canary = judge_one(judge_llm, todo[0])
    if canary and canary.get("error") == "llm unavailable":
        print(
            f"\nABORTING: judge model '{JUDGE_MODEL}' unreachable on the "
            "first call. Check UNI_API_KEY / backend status before "
            "re-running. No rows were written for this run.",
            file=sys.stderr,
        )
        sys.exit(1)
    print("Canary OK.\n", flush=True)

    fieldnames = [
        "file", "strategy_adherence", "deception_quality", "crew_defense",
        "outcome", "notes", "error",
    ]
    write_header = not os.path.exists(OUT_CSV)
    out = open(OUT_CSV, "a", newline="", buffering=1)
    writer = csv.DictWriter(out, fieldnames=fieldnames)
    if write_header:
        writer.writeheader()

    def _write(row: dict) -> None:
        writer.writerow({k: row.get(k, "") for k in fieldnames})

    _write(canary)
    remaining = todo[1:]

    t0 = time.time()
    done = 1
    error_buckets: dict[str, int] = {}
    if canary.get("error"):
        error_buckets[_error_bucket(canary["error"])] = 1

    with ThreadPoolExecutor(max_workers=PARALLELISM) as ex:
        futures = {ex.submit(judge_one, judge_llm, f): f for f in remaining}
        for fut in as_completed(futures):
            f = futures[fut]
            try:
                row = fut.result() or {"file": f, "error": "no result"}
            except Exception as e:
                row = {"file": f, "error": str(e)}
            _write(row)
            done += 1
            elapsed = (time.time() - t0) / 60.0
            msg = f"[{done}/{len(todo)}] {f}  elapsed={elapsed:.1f}m"
            if row.get("error"):
                bucket = _error_bucket(row["error"])
                error_buckets[bucket] = error_buckets.get(bucket, 0) + 1
                msg += f"  ERROR: {row['error'][:80]}"
            else:
                msg += (
                    f"  adh={row.get('strategy_adherence')} "
                    f"decep={row.get('deception_quality')} "
                    f"crew={row.get('crew_defense')}"
                )
            print(msg, flush=True)

    out.close()

    total_errors = sum(error_buckets.values())
    print(f"\nDone in {(time.time() - t0) / 60.0:.1f} min. Wrote {OUT_CSV}")
    if total_errors:
        print(f"Errors: {total_errors}/{done} rows — {error_buckets}")

    unavailable = error_buckets.get("llm unavailable", 0)
    if unavailable or total_errors > done / 2:
        print(
            f"\n*** WARNING: {unavailable} row(s) failed with 'llm "
            f"unavailable' and {total_errors}/{done} rows errored overall. "
            f"Scores in {OUT_CSV} may be incomplete/unreliable — check the "
            "judge backend before trusting this run. ***",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
