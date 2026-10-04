from abc import ABC, abstractmethod
from typing import Any, List

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from among_them.game.models.usage_metadata import UsageMetadata


class Agent(ABC, BaseModel):
    llm: ChatOpenAI = None
    responses: List[str] = Field(default_factory=list)
    player_name: str = ""
    role: str = ""
    token_usage: UsageMetadata = Field(default_factory=UsageMetadata)
    # When True the agent will retrieve curated strategies and past winning
    # plays from the RAG store and inject them into its prompts. Defaults to
    # False so unmodified games behave exactly as before.
    use_rag: bool = False
    # Per-strategy ablation: when this is non-empty, it is injected
    # verbatim into the prompt in place of any RAG-retrieved context. The
    # game then tests ONE strategy in isolation per game. Takes precedence
    # over ``use_rag`` so the caller can disable retrieval entirely.
    injected_strategy: str = ""

    @abstractmethod
    def act(self) -> Any:
        pass

    def add_token_usage(self, msg: dict):
        self.token_usage.input_tokens += msg["input_tokens"]
        self.token_usage.output_tokens += msg["output_tokens"]
        self.token_usage.total_tokens += msg["total_tokens"]
        if "cache_read" in msg["input_token_details"]:
            self.token_usage.cache_read += msg["input_token_details"]["cache_read"]

    def to_dict(self):
        llm_data = "human"
        if self.llm:
            if hasattr(self.llm, "model_name"):
                llm_data = self.llm.model_name
            elif hasattr(self.llm, "model"):
                llm_data = self.llm.model
            else:
                llm_data = "unknown"

        return {
            "llm": llm_data,
            "responses": self.responses,
            "player_name": self.player_name,
            "role": self.role,
            "token_usage": self.token_usage.to_dict(),
        }
