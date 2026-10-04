import os

from dotenv import load_dotenv

# Always load from .env first, this will override any existing environment variables
load_dotenv(override=True)

# Retrieve API keys and raise an error if they are missing
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

if not OPENROUTER_API_KEY:
    OPENROUTER_API_KEY = "None"
    # raise ValueError(
    #     "API key is missing. Please set OPENROUTER_API_KEY "
    #     "in your environment or in a .env file in the project root."
    # )

# Base URL for a local Ollama instance exposing an OpenAI-compatible API.
# Override with the OLLAMA_BASE_URL env var when running Ollama elsewhere.
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")

# University of Passau InnKube LLM hub (OpenAI-compatible).
# Set UNI_API_KEY in .env; UNI_BASE_URL can be overridden if the hub moves.
UNI_API_KEY = os.getenv("UNI_API_KEY")
UNI_BASE_URL = os.getenv(
    "UNI_BASE_URL", "https://llms.innkube.fim.uni-passau.de/v1"
)

# Directory for new tournament / batch game JSON outputs.
# Override with RUNS_DIR in .env if you want a different location.
RUNS_DIR = os.getenv("RUNS_DIR", "data/runs")
os.makedirs(RUNS_DIR, exist_ok=True)

# if src/among_them/game/dummy.py does not exist, create it.
# This is for abusing streamlit refresh when game_state.json changes
if not os.path.exists("src/among_them/game/dummy.py"):
    with open("src/among_them/game/dummy.py", "w") as f:
        f.write("timestamp = '2024-11-15 00:20:17.946790'")

# Models that automatically get RAG boost at game start.
# These are the weaker / smaller models that benefit most from retrieved
# context (curated strategies + winning plays from prior games). The
# Streamlit picker uses this set to default the per-team RAG checkbox.
RAG_BOOSTED_MODELS: set[str] = {
    "uni/qwen36-35b",
    "uni/gemma4-31b-it",
    "ollama/gemma3:1b",
    "ollama/llama3.2:1b",
    "openai/gpt-4o-mini",
    "meta-llama/llama-3.1-8b-instruct",
}
