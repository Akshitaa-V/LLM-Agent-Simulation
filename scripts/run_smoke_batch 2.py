"""Small ad-hoc batch runner for smoke games on InnKube models.

Runs N games sequentially and dumps JSONs into ``data/runs/`` with
explicit filenames so they don't clash with the teammate's runs. Kept
out of the main tournament flow so we can iterate without touching it.
"""

import os
import sys
import time
import traceback

# Make ``among_them`` importable when run directly.
HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.abspath(os.path.join(HERE, "..", "src"))
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from among_them.config import RUNS_DIR  # noqa: E402
from among_them.tournament import TournamentGame  # noqa: E402

OUT_DIR = RUNS_DIR
ROUND_CAP = 20
N_PLAYERS = 5
N_IMPOSTORS = 1

GAMES = [
    # (impostor_model, crewmate_model, filename)
    (
        "uni/qwen3-next-80b-a3b-instruct",
        "uni/qwen3-next-80b-a3b-instruct",
        "exp_smoke_qwen3next_vs_qwen3next_1.json",
    ),
    (
        "uni/qwen3-next-80b-a3b-instruct",
        "uni/qwen36-35b",
        "exp_smoke_qwen3next_vs_qwen36_1.json",
    ),
    (
        "uni/qwen36-35b",
        "uni/qwen3-next-80b-a3b-instruct",
        "exp_smoke_qwen36_vs_qwen3next_1.json",
    ),
    (
        "uni/gemma4-31b-it",
        "uni/qwen36-35b",
        "exp_smoke_gemma_vs_qwen36_1.json",
    ),
]


def main() -> None:
    project_root = os.path.abspath(os.path.join(HERE, ".."))
    os.chdir(project_root)
    os.makedirs(OUT_DIR, exist_ok=True)

    batch_start = time.time()
    for idx, (imp, crew, fname) in enumerate(GAMES, start=1):
        print(f"\n{'=' * 70}")
        print(f"[{idx}/{len(GAMES)}] {imp} (imp) vs {crew} (crew) -> {fname}")
        print(f"{'=' * 70}", flush=True)
        t0 = time.time()
        try:
            game = TournamentGame(
                n_players=N_PLAYERS,
                n_impostors=N_IMPOSTORS,
                impostor_model=imp,
                crewmate_model=crew,
                dir=OUT_DIR,
                file_path=fname,
                n_round_cut_off=ROUND_CAP,
                debug=False,
            )
            game.run()
            dt = time.time() - t0
            print(f"[{idx}/{len(GAMES)}] done in {dt / 60:.1f} min", flush=True)
        except Exception as e:
            print(f"[{idx}/{len(GAMES)}] FAILED: {type(e).__name__}: {e}")
            traceback.print_exc()

    total = time.time() - batch_start
    print(f"\nBatch finished in {total / 60:.1f} min")


if __name__ == "__main__":
    main()
