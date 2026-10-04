"""Batch of N RAG-on-impostor games for accumulating statistics.

Reuses the per-game logic in run_rag_compare.py — same matchup, same
RAG configuration, just looped with sequential rep numbers so output
files don't clash.
"""

import os
import sys
import time
import traceback
from random import shuffle

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.abspath(os.path.join(HERE, "..", "src"))
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from among_them.config import RUNS_DIR  # noqa: E402
from among_them.game.game_engine import GameEngine  # noqa: E402
from among_them.game.models.engine import GamePhase  # noqa: E402
from among_them.game.players.ai import AIPlayer  # noqa: E402

IMPOSTOR_MODEL = "uni/qwen36-35b"
CREWMATE_MODEL = "uni/qwen3-next-80b-a3b-instruct"
N_PLAYERS = 5
N_IMPOSTORS = 1
ROUND_CAP = 20

# Reps to run. Previous batches produced _1 .. _11. This batch adds
# 15 more reps (_12 .. _26) using the expanded strategy set + indexed
# past-game plays for a tighter sample size.
REPS = list(range(12, 27))

NAME_TEMPLATE = "exp_rag_qwen36rag_vs_qwen3next_{rep}.json"


def build_players():
    names = ["Alice", "Bob", "Charlie", "Dave", "Eren"][:N_PLAYERS]
    shuffle(names)
    players = []
    for i in range(N_PLAYERS):
        if i < N_IMPOSTORS:
            players.append(
                AIPlayer(
                    name=names[i],
                    llm_model_name=IMPOSTOR_MODEL,
                    role="Impostor",
                    use_rag=True,
                )
            )
        else:
            players.append(
                AIPlayer(
                    name=names[i],
                    llm_model_name=CREWMATE_MODEL,
                    role="Crewmate",
                    use_rag=False,
                )
            )
    return players


def add_flag(full_path: str, flag: str) -> None:
    new_path = full_path.split(".")[0] + "_" + flag + ".json"
    if os.path.exists(full_path):
        os.rename(full_path, new_path)


def run_one(out_file: str) -> dict:
    full_path = os.path.join(RUNS_DIR, out_file)
    players = build_players()
    engine = GameEngine(file_path=full_path)
    engine.state.DEBUG = False
    engine.load_players(players, impostor_count=N_IMPOSTORS)
    engine.state.game_stage = GamePhase.ACTION_PHASE

    t0 = time.time()
    finished = False
    err = None
    while not finished and engine.state.round_number < ROUND_CAP:
        try:
            finished = engine.perform_step()
        except Exception as e:
            if "LLM did" in str(e):
                continue
            err = e
            try:
                add_flag(full_path, "exception")
            except Exception:
                pass
            with open("data/error_log.txt", "a") as f:
                f.write(f"{out_file} errored with:\n{e}\n")
            traceback.print_exc()
            break

    if not err and engine.state.round_number >= ROUND_CAP:
        add_flag(full_path, "round_limit")

    return {
        "file": out_file,
        "elapsed_min": (time.time() - t0) / 60.0,
        "rounds": engine.state.round_number,
        "error": str(err) if err else None,
    }


def main() -> None:
    project_root = os.path.abspath(os.path.join(HERE, ".."))
    os.chdir(project_root)
    os.makedirs(RUNS_DIR, exist_ok=True)

    print(f"Batch size: {len(REPS)} games")
    print(f"Impostor:   {IMPOSTOR_MODEL}  use_rag=True")
    print(f"Crewmate:   {CREWMATE_MODEL}  use_rag=False")
    print(f"Round cap:  {ROUND_CAP}")
    print(f"Out dir:    {RUNS_DIR}\n", flush=True)

    batch_start = time.time()
    results = []
    for i, rep in enumerate(REPS, start=1):
        out_file = NAME_TEMPLATE.format(rep=rep)
        print(f"{'=' * 70}")
        print(f"[{i}/{len(REPS)}] -> {out_file}")
        print(f"{'=' * 70}", flush=True)
        r = run_one(out_file)
        results.append(r)
        msg = (
            f"[{i}/{len(REPS)}] done rounds={r['rounds']} "
            f"in {r['elapsed_min']:.1f} min"
        )
        if r["error"]:
            msg += f"  ERROR: {r['error']}"
        print(msg + "\n", flush=True)

    total = (time.time() - batch_start) / 60.0
    print(f"\nBatch finished in {total:.1f} min total")
    print(f"Avg per game: {total / len(REPS):.1f} min")


if __name__ == "__main__":
    main()
