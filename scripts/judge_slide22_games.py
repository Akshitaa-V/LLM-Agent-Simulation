"""Judge the two slide-22 one-off sanity-check games only.

Companion to ``judge_games.py``, scoped to the two ad-hoc scale-check
games that aren't part of the main 228-game batch and so never got
picked up by the regular judge run:

- exp_bigger_gemma_vs_qwen3next_post_kill_alibi_1.json
    (gemma4-31b impostor vs. qwen3-next-80b crew -- "bigger crew" check)
- exp_80b_vs_gemma_meta_1.json
    (qwen3-next-80b impostor vs. gemma4-31b crew, Meta Crew -- "bigger
    impostor" check)

Both were run with the post_kill_alibi strategy injected, but that
strategy name isn't recoverable from the filename the way it is for
the main batch (those follow exp_big_*_<strategy>_<rep>.json), so
it's hardcoded below instead of guessed.

Run from the project root:

    python scripts/judge_slide22_games.py

Writes to data/exp_judge_scores_slide22.csv (kept separate from the
main data/exp_judge_scores.csv so this ad-hoc run can't collide with
or get mistaken for the batch-scored numbers).
"""

import csv
import json
import os
import sys

from langchain.schema import HumanMessage, SystemMessage

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.abspath(os.path.join(HERE, "..", "src"))
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from among_them.config import RUNS_DIR  # noqa: E402
from among_them.llm_factory import make_llm, safe_invoke  # noqa: E402
from among_them.rag.seeder import CURATED_STRATEGIES  # noqa: E402

# Reuse judge_games.py's exact prompt/scoring so the numbers are
# directly comparable to the main 228-game batch.
from judge_games import SYSTEM_PROMPT, build_user_prompt, parse_scores  # noqa: E402

_STRATEGY_TEXT = {s["name"]: s["text"] for s in CURATED_STRATEGIES}

JUDGE_MODEL = "uni/qwen3-next-80b-a3b-instruct"
OUT_CSV = "data/exp_judge_scores_slide22.csv"

# file_name -> injected strategy name (edit here if either game used a
# different strategy than assumed).
FILES = {
    "exp_bigger_gemma_vs_qwen3next_post_kill_alibi_1.json": "post_kill_alibi",
    "exp_80b_vs_gemma_meta_1.json": "post_kill_alibi",
}


def extract_inputs(game_path: str, strategy_name: str) -> dict | None:
    with open(game_path) as f:
        s = json.load(f)
    players = s["players"]
    impostor_names = [p["name"] for p in players if p.get("is_impostor")]
    if not impostor_names:
        return None

    strategy = _STRATEGY_TEXT.get(strategy_name, "")

    utterances = []
    for line in s.get("playthrough", []):
        if "chat" not in line:
            continue
        import re
        m = re.search(r"\[([^\]]+)\]:\s*(.+)$", line)
        if m:
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


def judge_one(judge_llm, file_name: str, strategy_name: str) -> dict:
    path = os.path.join(RUNS_DIR, file_name)
    try:
        inputs = extract_inputs(path, strategy_name)
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


def main() -> None:
    project_root = os.path.abspath(os.path.join(HERE, ".."))
    os.chdir(project_root)

    print(f"Judge model: {JUDGE_MODEL}")
    print(f"Files to score: {list(FILES)}\n", flush=True)

    judge_llm = make_llm(JUDGE_MODEL, temperature=0)

    fieldnames = [
        "file", "strategy_adherence", "deception_quality", "crew_defense",
        "outcome", "notes", "error",
    ]
    rows = []
    for file_name, strategy_name in FILES.items():
        print(f"Judging {file_name} (strategy={strategy_name})...", flush=True)
        row = judge_one(judge_llm, file_name, strategy_name)
        rows.append(row)
        if row.get("error"):
            print(f"  ERROR: {row['error']}", flush=True)
        else:
            print(
                f"  adherence={row['strategy_adherence']} "
                f"deception={row['deception_quality']} "
                f"crew_defense={row['crew_defense']} "
                f"outcome={row['outcome']}",
                flush=True,
            )

    with open(OUT_CSV, "w", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})

    print(f"\nWrote {OUT_CSV}")
    if any(r.get("error") for r in rows):
        print(
            "\n*** Some rows errored -- check UNI_API_KEY / backend before "
            "trusting these numbers. ***",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
