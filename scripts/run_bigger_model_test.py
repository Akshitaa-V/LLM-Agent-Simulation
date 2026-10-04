"""One-off: gemma4-31b impostor (post_kill_alibi) vs qwen3-next-80b crew.

Same 10p/2i/40-round setup as the big batch. Purpose is to sanity-check
whether the gemma-impostor dominance survives against a larger crewmate
model.
"""

import os
import subprocess
import sys
import time
import traceback
from random import Random

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
CREWMATE_MODEL = "uni/qwen3-next-80b-a3b-instruct"
N_PLAYERS = 10
N_IMPOSTORS = 2
ROUND_CAP = 40
STRATEGY_NAME = "post_kill_alibi"
OUT_FILE = "exp_bigger_gemma_vs_qwen3next_post_kill_alibi_1.json"


def main() -> None:
    project_root = os.path.abspath(os.path.join(HERE, ".."))
    os.chdir(project_root)
    os.makedirs(RUNS_DIR, exist_ok=True)

    strat = next(
        s for s in CURATED_STRATEGIES if s["name"] == STRATEGY_NAME
    )
    imp_text = strat["text"]

    rng = Random(42)
    names = [
        "Alice", "Bob", "Charlie", "Dave", "Eren",
        "Modi", "Macroon", "Doraemon", "Ian", "Judy",
    ]
    rng.shuffle(names)

    players = []
    for i in range(N_PLAYERS):
        if i < N_IMPOSTORS:
            players.append(AIPlayer(
                name=names[i],
                llm_model_name=IMPOSTOR_MODEL,
                role="Impostor",
                use_rag=False,
                injected_strategy=imp_text,
            ))
        else:
            players.append(AIPlayer(
                name=names[i],
                llm_model_name=CREWMATE_MODEL,
                role="Crewmate",
                use_rag=False,
                injected_strategy="",
            ))

    full_path = os.path.join(RUNS_DIR, OUT_FILE)
    engine = GameEngine(file_path=full_path)
    engine.state.DEBUG = False
    engine.load_players(players, impostor_count=N_IMPOSTORS)
    engine.state.game_stage = GamePhase.ACTION_PHASE

    print(f"Impostor: {IMPOSTOR_MODEL}  (strategy: {STRATEGY_NAME})")
    print(f"Crewmate: {CREWMATE_MODEL}")
    print(f"{N_PLAYERS} players, {N_IMPOSTORS} impostor, cap {ROUND_CAP}")
    print(f"Output: {OUT_FILE}\n", flush=True)

    t0 = time.time()
    finished = False
    err = None
    MAX_CONSECUTIVE_RETRIES = 5
    consecutive_retries = 0
    while not finished and engine.state.round_number < ROUND_CAP:
        try:
            finished = engine.perform_step()
            consecutive_retries = 0
        except Exception as e:
            if "LLM did" in str(e):
                consecutive_retries += 1
                print(
                    f"[retry {consecutive_retries}/{MAX_CONSECUTIVE_RETRIES}] "
                    f"non-conforming LLM output, retrying: {e}",
                    flush=True,
                )
                if consecutive_retries >= MAX_CONSECUTIVE_RETRIES:
                    err = e
                    print(
                        f"ABORTING: {MAX_CONSECUTIVE_RETRIES} consecutive "
                        "non-conforming turns, giving up.",
                        flush=True,
                    )
                    break
                continue
            err = e
            traceback.print_exc()
            break

    if not err and engine.state.round_number >= ROUND_CAP:
        new_path = full_path.replace(".json", "_round_limit.json")
        if os.path.exists(full_path):
            os.rename(full_path, new_path)

    elapsed = (time.time() - t0) / 60.0
    print(
        f"\nFinished in {elapsed:.1f} min. Rounds={engine.state.round_number}",
        flush=True,
    )
    if err:
        print(f"ERROR: {err}", flush=True)

    print("Regenerating analyzer CSVs...", flush=True)
    try:
        subprocess.run(
            [sys.executable, os.path.join(HERE, "analyze_runs.py")],
            check=True,
        )
    except subprocess.CalledProcessError as e:
        print(f"Analyzer exited with {e.returncode}", file=sys.stderr, flush=True)
    except FileNotFoundError as e:
        print(f"Analyzer script not found: {e}", file=sys.stderr, flush=True)

    if err:
        sys.exit(1)


if __name__ == "__main__":
    main()
