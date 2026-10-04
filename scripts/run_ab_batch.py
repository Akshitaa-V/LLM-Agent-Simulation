"""Matched A/B batch: 10 no-RAG impostor + 10 RAG impostor.

Holds everything constant — same impostor model (``uni/qwen36-35b``),
same crewmate model (``uni/qwen3-next-80b-a3b-instruct``), same player
count, same round cap — and only flips ``use_rag`` on the impostor.

Output files:
* ``exp_norag_qwen36_vs_qwen3next_1.json`` .. ``_10.json``
* ``exp_rag_qwen36rag_vs_qwen3next_27.json`` .. ``_36.json``
  (continues the existing rep numbering so the analyzer's existing
  ``exp_rag_`` prefix logic catches them automatically.)

The RAG store is whatever ``seed_strategies()`` plus
``seed_from_game_logs()`` have already populated — this script does
NOT re-seed before running, so the comparison uses the current store
state (18 curated strategies + indexed past plays).
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

NO_RAG_REPS = list(range(1, 11))
RAG_REPS = list(range(27, 37))

NO_RAG_NAME = "exp_norag_qwen36_vs_qwen3next_{rep}.json"
RAG_NAME = "exp_rag_qwen36rag_vs_qwen3next_{rep}.json"


def build_players(impostor_use_rag: bool):
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
                    use_rag=impostor_use_rag,
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


def run_one(out_file: str, impostor_use_rag: bool) -> dict:
    full_path = os.path.join(RUNS_DIR, out_file)
    players = build_players(impostor_use_rag)
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


def run_block(label: str, template: str, reps: list[int], use_rag: bool, batch_t0: float) -> None:
    print(f"\n{'#' * 70}")
    print(f"# BLOCK: {label}  ({len(reps)} games, use_rag={use_rag})")
    print(f"{'#' * 70}", flush=True)
    for i, rep in enumerate(reps, start=1):
        out_file = template.format(rep=rep)
        print(f"\n[{label} {i}/{len(reps)}] -> {out_file}", flush=True)
        r = run_one(out_file, impostor_use_rag=use_rag)
        msg = (
            f"[{label} {i}/{len(reps)}] done rounds={r['rounds']} "
            f"in {r['elapsed_min']:.1f} min  "
            f"(batch elapsed: {(time.time() - batch_t0) / 60.0:.1f} min)"
        )
        if r["error"]:
            msg += f"  ERROR: {r['error']}"
        print(msg, flush=True)


def main() -> None:
    project_root = os.path.abspath(os.path.join(HERE, ".."))
    os.chdir(project_root)
    os.makedirs(RUNS_DIR, exist_ok=True)

    print(f"A/B batch: {len(NO_RAG_REPS)} no-RAG + {len(RAG_REPS)} RAG = "
          f"{len(NO_RAG_REPS) + len(RAG_REPS)} games")
    print(f"Impostor:   {IMPOSTOR_MODEL}")
    print(f"Crewmate:   {CREWMATE_MODEL}")
    print(f"Round cap:  {ROUND_CAP}")
    print(f"Out dir:    {RUNS_DIR}", flush=True)

    batch_t0 = time.time()
    run_block("no-RAG", NO_RAG_NAME, NO_RAG_REPS, use_rag=False, batch_t0=batch_t0)
    run_block("RAG", RAG_NAME, RAG_REPS, use_rag=True, batch_t0=batch_t0)

    total = (time.time() - batch_t0) / 60.0
    n_total = len(NO_RAG_REPS) + len(RAG_REPS)
    print(f"\nBatch finished in {total:.1f} min total")
    print(f"Avg per game: {total / n_total:.1f} min")


if __name__ == "__main__":
    main()
