from typing import List

from langchain.schema import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import Field

from among_them.llm_factory import make_llm, safe_invoke
from among_them.llm_prompts import VOTING_SYSTEM_PROMPT, VOTING_USER_PROMPT
from among_them.game.utils import check_action_valid
from among_them.rag.retriever import get_voting_context

from .base_agent import Agent


class VotingAgent(Agent):
    llm: ChatOpenAI = None
    llm_model_name: str
    history: str = Field(default="")
    available_actions: List[str] = Field(default_factory=list)

    def __init__(self, **data):
        super().__init__(**data)
        self.init_llm()

    def init_llm(self):
        self.llm = make_llm(self.llm_model_name, temperature=0)

    def act(
        self,
        observations: str,
        actions: List[str],
        discussion_log: str,
        dead_players: List[str],
    ) -> int:
        self.history = observations
        self.available_actions = actions

        rag_context = ""
        if self.injected_strategy:
            rag_context = (
                "--- Strategy for your role ---\n" + self.injected_strategy
            )
        elif self.use_rag:
            rag_context = get_voting_context(
                role=self.role,
                discussion_log=discussion_log,
            )

        system_prompt = VOTING_SYSTEM_PROMPT

        user_prompt = VOTING_USER_PROMPT.format(
            rag_context=rag_context,
            player_name=self.player_name,
            player_role=self.role,
            discussion_log=discussion_log,
            history=self.history,
            actions="\n".join(f"- {action}" for action in self.available_actions),
            dead_players=", ".join(dead_players),
        )

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]

        # Small local models often emit malformed votes, and the upstream
        # engine can be transiently unavailable. Retry up to 3 times for
        # either failure mode; if still invalid, fall back to "vote for nobody".
        last_text = ""
        for attempt in range(3):
            chosen_action = safe_invoke(
                self.llm,
                messages,
                attempts=1,
                label=f"vote/{self.player_name}",
            )
            if chosen_action is None:
                continue
            self.add_token_usage(chosen_action.usage_metadata)
            last_text = chosen_action.content.strip()
            try:
                vote_idx, _ = check_action_valid(
                    self.available_actions, last_text, self.player_name
                )
                self.responses.append(last_text)
                return [user_prompt], vote_idx
            except ValueError:
                print(
                    f"\033[33m[VotingAgent] {self.player_name} bad output "
                    f"(attempt {attempt + 1}/3): {last_text!r}\033[0m"
                )

        fallback_idx = self._safe_vote_fallback_index()
        fallback_text = self.available_actions[fallback_idx]
        print(
            f"\033[33m[VotingAgent] {self.player_name} falling back to "
            f"{fallback_text!r}\033[0m"
        )
        self.responses.append(f"[fallback after 3 retries] {last_text}")
        return [user_prompt], fallback_idx

    def _safe_vote_fallback_index(self) -> int:
        for i, action in enumerate(self.available_actions):
            if "nobody" in action.lower():
                return i
        return 0
