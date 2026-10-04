"""Factory for building ChatOpenAI clients pointed at the right backend.

Three backends are supported via simple model-name prefixes:

* ``ollama/<tag>`` -> local Ollama instance (OpenAI-compatible API).
* ``uni/<id>``    -> University of Passau InnKube LLM hub
                       (OpenAI-compatible API, UNI_API_KEY required).
* anything else   -> OpenRouter (OPENROUTER_API_KEY required).

This keeps the rest of the code base unchanged: agents just pass a model
name string and ``make_llm`` decides where to send the request.
"""

import time
from typing import List, Optional

from langchain.schema import BaseMessage
from langchain_openai import ChatOpenAI

from among_them.config import (
    OLLAMA_BASE_URL,
    OPENROUTER_API_KEY,
    UNI_API_KEY,
    UNI_BASE_URL,
)

OLLAMA_PREFIX = "ollama/"
UNI_PREFIX = "uni/"

# Per-call timeout in seconds. Without this the underlying httpx client
# can sit on a hung socket forever after a system sleep or upstream
# drop. 180s is generous for slow uni/* responses but bounded.
LLM_TIMEOUT = 180.0


def safe_invoke(
    llm: ChatOpenAI,
    messages: List[BaseMessage],
    *,
    attempts: int = 3,
    backoff: float = 1.5,
    label: str = "",
):
    """Invoke ``llm`` with retries on any exception.

    Returns the AIMessage on success or ``None`` after exhausting ``attempts``.
    Useful when the upstream model engine is flaky (e.g. an InnKube vLLM pod
    is restarting) -- a transient failure shouldn't crash the whole game.
    """
    last_error: Optional[BaseException] = None
    for i in range(attempts):
        try:
            return llm.invoke(messages)
        except Exception as e:  # noqa: BLE001 - intentionally broad
            last_error = e
            msg = str(e).splitlines()[0][:200]
            print(
                f"\033[33m[safe_invoke] {label or 'LLM call'} attempt "
                f"{i + 1}/{attempts} failed: {msg}\033[0m"
            )
            if i < attempts - 1:
                time.sleep(backoff * (i + 1))
    print(
        f"\033[31m[safe_invoke] {label or 'LLM call'} giving up after "
        f"{attempts} attempts ({type(last_error).__name__}).\033[0m"
    )
    return None


def make_llm(model_name: str, temperature: float) -> ChatOpenAI:
    """Build a ChatOpenAI client for ``model_name``."""
    if model_name.startswith(OLLAMA_PREFIX):
        return ChatOpenAI(
            base_url=OLLAMA_BASE_URL,
            api_key="ollama",
            model=model_name.removeprefix(OLLAMA_PREFIX),
            temperature=temperature,
            timeout=LLM_TIMEOUT,
            max_retries=1,
        )

    if model_name.startswith(UNI_PREFIX):
        if not UNI_API_KEY:
            raise ValueError(
                "Missing University LLM hub API key. "
                "Please set UNI_API_KEY in your environment or .env file."
            )
        return ChatOpenAI(
            base_url=UNI_BASE_URL,
            api_key=UNI_API_KEY,
            model=model_name.removeprefix(UNI_PREFIX),
            temperature=temperature,
            timeout=LLM_TIMEOUT,
            max_retries=1,
        )

    if not OPENROUTER_API_KEY or OPENROUTER_API_KEY == "None":
        raise ValueError(
            "Missing OpenRouter API key. "
            "Please set OPENROUTER_API_KEY in your environment."
        )

    return ChatOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY,
        model=model_name,
        temperature=temperature,
        timeout=LLM_TIMEOUT,
        max_retries=1,
    )
