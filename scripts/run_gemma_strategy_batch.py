"""Run 10 games with gemma impostor vs qwen36 crewmates.

Each game injects one impostor strategy from CURATED_STRATEGIES (the RAG
seed list) verbatim into the impostor prompts. Files land in
``data/runs/exp_gemma_strat_<strategy>_1.json`` so the analyzer groups
them under their strategy name.

After all games finish, the run-analyzer is re-invoked to regenerate the
CSVs the GUI reads.
"""

import os
import subprocess
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
from among_them.rag.seeder import CURATED_STRATEGIES  # noqa: E402

IMPOSTOR_MODEL = "uni/gemma4-31b-it"
CREWMATE_MODEL = "uni/qwen36-35b"
N_PLAYERS = 5
N_IMPOSTORS = 1
ROUND_CAP = 20
N_GAMES = 10


def build_players(injected_strategy: str):
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
                    use_rag=False,
                    injected_strategy=injected_strategy,
                )
            )
        else:
            players.append(
                AIPlayer(
                    name=names[i],
                    llm_model_name=CREWMATE_MODEL,
                    role="Crewmate",
                    use_rag=False,
                    injected_strategy="",
                )
            )
    return players


def add_flag(full_path: str, flag: str) -> None:
    new_path = full_path.split(".")[0] + "_" + flag + ".json"
    if os.path.exists(full_path):
        os.rename(full_path, new_path)


def run_one(out_file: str, strategy_text: str) -> dict:
    full_path = os.path.join(RUNS_DIR, out_file)
    players = build_players(strategy_text)
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

    impostor_strats = [
        s for s in CURATED_STRATEGIES if s.get("role", "").lower() == "impostor"
    ][:N_GAMES]
    if len(impostor_strats) < N_GAMES:
        print(
            f"WARNING: only {len(impostor_strats)} impostor strategies available,"
            f" wanted {N_GAMES}",
            flush=True,
        )

    print(f"Impostor: {IMPOSTOR_MODEL}")
    print(f"Crewmate: {CREWMATE_MODEL}")
    print(f"Games: {len(impostor_strats)}, round cap: {ROUND_CAP}\n", flush=True)

    batch_t0 = time.time()
    for i, strat in enumerate(impostor_strats, start=1):
        name = strat["name"]
        out_file = f"exp_gemma_strat_{name}_1.json"
        print(
            f"[{i}/{len(impostor_strats)}] strategy={name} -> {out_file}",
            flush=True,
        )
        r = run_one(out_file, strat["text"])
        msg = (
            f"  done rounds={r['rounds']} in {r['elapsed_min']:.1f} min "
            f"(batch {(time.time() - batch_t0) / 60.0:.1f} min)"
        )
        if r["error"]:
            msg += f"  ERROR: {r['error']}"
        print(msg, flush=True)

    print(f"\nBatch finished in {(time.time() - batch_t0) / 60.0:.1f} min")

    print("Regenerating analyzer CSVs...", flush=True)
    try:
        subprocess.run(
            [sys.executable, os.path.join(HERE, "analyze_runs.py")],
            check=True,
        )
    except Exception as e:
        print(f"Analyzer failed: {e}", flush=True)


if __name__ == "__main__":
    main()
