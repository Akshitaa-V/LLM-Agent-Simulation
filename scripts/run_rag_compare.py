"""Run a single game with RAG enabled only on the smaller impostor model.

Designed as a direct A/B against ``exp_smoke_qwen36_vs_qwen3next_1.json``
from the smoke batch: same matchup, same round cap, same player count.
The only difference is ``use_rag=True`` on the impostor.

Mirrors ``TournamentGame`` from ``among_them.tournament`` but constructs
the player list manually so we can flip ``use_rag`` per side.
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
OUT_FILE = "exp_rag_qwen36rag_vs_qwen3next_1.json"


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
                    use_rag=True,  # <- the experiment
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


def main() -> None:
    project_root = os.path.abspath(os.path.join(HERE, ".."))
    os.chdir(project_root)
    os.makedirs(RUNS_DIR, exist_ok=True)
    full_path = os.path.join(RUNS_DIR, OUT_FILE)

    print(f"Impostor: {IMPOSTOR_MODEL} (use_rag=True)")
    print(f"Crewmate: {CREWMATE_MODEL} (use_rag=False)")
    print(f"Output:   {full_path}")
    print(f"Round cap: {ROUND_CAP}\n", flush=True)

    players = build_players()
    engine = GameEngine(file_path=full_path)
    engine.state.DEBUG = False
    engine.load_players(players, impostor_count=N_IMPOSTORS)
    engine.state.game_stage = GamePhase.ACTION_PHASE

    t0 = time.time()
    finished = False
    while not finished and engine.state.round_number < ROUND_CAP:
        try:
            finished = engine.perform_step()
        except Exception as e:
            if "LLM did" in str(e):
                continue
            try:
                add_flag(full_path, "exception")
            except Exception:
                pass
            with open("data/error_log.txt", "a") as f:
                f.write(f"{OUT_FILE} errored with:\n{e}\n")
            traceback.print_exc()
            break

    if engine.state.round_number >= ROUND_CAP:
        add_flag(full_path, "round_limit")

    dt = time.time() - t0
    print(f"\nDone in {dt / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
