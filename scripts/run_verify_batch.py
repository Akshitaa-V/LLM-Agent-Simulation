"""Quick 5-game verification batch.

Sanity-checks the full pipeline (LLM clients, game engine, save/load,
the post-venv-rebuild code paths) by running 5 games with the standard
qwen36-35b (impostor, no RAG) vs qwen3-next-80b (crewmates) matchup.
Files saved as ``data/runs/exp_verify_<rep>.json`` so they're easy to
identify and aren't conflated with the A/B or ablation sets.
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
N_REPS = 5
NAME_TEMPLATE = "exp_verify_{rep}.json"


def build_players():
    names = ["Alice", "Bob", "Charlie", "Dave", "Eren"][:N_PLAYERS]
    shuffle(names)
    return [
        AIPlayer(
            name=names[i],
            llm_model_name=IMPOSTOR_MODEL if i < N_IMPOSTORS else CREWMATE_MODEL,
            role="Impostor" if i < N_IMPOSTORS else "Crewmate",
            use_rag=False,
            injected_strategy="",
        )
        for i in range(N_PLAYERS)
    ]


def add_flag(full_path: str, flag: str) -> None:
    new_path = full_path.split(".")[0] + "_" + flag + ".json"
    if os.path.exists(full_path):
        os.rename(full_path, new_path)


def run_one(out_file: str) -> dict:
    full_path = os.path.join(RUNS_DIR, out_file)
    engine = GameEngine(file_path=full_path)
    engine.state.DEBUG = False
    engine.load_players(build_players(), impostor_count=N_IMPOSTORS)
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

    # Classify winner from the playthrough.
    pt = engine.state.playthrough
    if any("Crewmates win!" in line for line in pt):
        winner = "Crewmates"
    elif any("Impostors win!" in line for line in pt):
        winner = "Impostors"
    else:
        winner = "Unfinished"

    return {
        "file": out_file,
        "elapsed_min": (time.time() - t0) / 60.0,
        "rounds": engine.state.round_number,
        "winner": winner,
        "error": str(err) if err else None,
    }


def main() -> None:
    project_root = os.path.abspath(os.path.join(HERE, ".."))
    os.chdir(project_root)
    os.makedirs(RUNS_DIR, exist_ok=True)

    print(f"Verification batch: {N_REPS} games")
    print(f"Impostor:  {IMPOSTOR_MODEL}")
    print(f"Crewmate:  {CREWMATE_MODEL}")
    print(f"Round cap: {ROUND_CAP}")
    print(f"Out dir:   {RUNS_DIR}\n", flush=True)

    batch_t0 = time.time()
    results = []
    for rep in range(1, N_REPS + 1):
        out_file = NAME_TEMPLATE.format(rep=rep)
        print(f"\n[{rep}/{N_REPS}] -> {out_file}", flush=True)
        r = run_one(out_file)
        results.append(r)
        msg = (
            f"[{rep}/{N_REPS}] done winner={r['winner']} "
            f"rounds={r['rounds']} in {r['elapsed_min']:.1f} min  "
            f"(batch elapsed: {(time.time() - batch_t0) / 60.0:.1f} min)"
        )
        if r["error"]:
            msg += f"  ERROR: {r['error']}"
        print(msg, flush=True)

    total = (time.time() - batch_t0) / 60.0
    print(f"\nBatch finished in {total:.1f} min total")

    # Quick verification summary
    decisive = [r for r in results if r["winner"] in ("Crewmates", "Impostors")]
    imp_wins = sum(1 for r in decisive if r["winner"] == "Impostors")
    crew_wins = sum(1 for r in decisive if r["winner"] == "Crewmates")
    caps = sum(1 for r in results if r["winner"] == "Unfinished")
    errors = sum(1 for r in results if r["error"])
    print(f"\n=== VERIFY SUMMARY ===")
    print(f"  Games:        {len(results)}")
    print(f"  Decisive:     {len(decisive)}  (imp_wins={imp_wins}, crew_wins={crew_wins})")
    print(f"  Round-caps:   {caps}")
    print(f"  Errors:       {errors}")
    if errors == 0 and caps <= 1:
        print("  Pipeline: OK")
    else:
        print("  Pipeline: check errors / round caps above")


if __name__ == "__main__":
    main()
