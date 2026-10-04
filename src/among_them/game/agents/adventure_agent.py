from typing import Any, List

from langchain.schema import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import Field

from among_them.game.agents.base_agent import Agent
from among_them.llm_factory import make_llm, safe_invoke
from among_them.llm_prompts import (
    ADVENTURE_ACTION_SYSTEM_PROMPT,
    ADVENTURE_ACTION_USER_PROMPT,
    ADVENTURE_PLAN_SYSTEM_PROMPT,
    ADVENTURE_PLAN_USER_PROMPT,
)
from among_them.game.utils import check_action_valid
from among_them.rag.retriever import get_action_context


class AdventureAgent(Agent):
    llm: ChatOpenAI = None
    response_llm: ChatOpenAI = None
    llm_model_name: str
    history: str = Field(default="")
    current_tasks: List[Any] = Field(default_factory=list)
    available_actions: List[str] = Field(default_factory=list)
    current_location: str = Field(default="")
    in_room: str = Field(default="")

    def __init__(self, **data):
        super().__init__(**data)
        self.init_llm()

    def init_llm(self):
        self.llm = make_llm(self.llm_model_name, temperature=1)
        self.response_llm = make_llm(self.llm_model_name, temperature=0)

    def act(
        self,
        observations: str,
        tasks: List[str],
        actions: List[str],
        current_location: str,
        in_room: str,
    ) -> Any:
        self.history = observations
        self.current_tasks = tasks
        self.available_actions = actions
        self.current_location = current_location
        self.in_room = in_room

        plan_prompt, plan = self.create_plan()
        action_prompt, action_idx, action = self.choose_action(plan)
        self.responses.append(plan)
        self.responses.append(action)
        return [plan_prompt, action_prompt], action_idx

    def create_plan(self) -> str:
        rag_context = ""
        if self.injected_strategy:
            # Single-strategy ablation: bypass retrieval and inject this
            # strategy verbatim so every prompt the impostor sees this
            # round contains the exact same strategy text.
            rag_context = (
                "--- Strategy for your role ---\n" + self.injected_strategy
            )
        elif self.use_rag:
            rag_context = get_action_context(
                role=self.role,
                location=self.current_location,
                in_room=self.in_room,
            )

        plan_prompt = ADVENTURE_PLAN_USER_PROMPT.format(
            rag_context=rag_context,
            player_name=self.player_name,
            player_role=self.role,
            history=self.history,
            tasks=[str(task) for task in self.current_tasks],
            actions="- " + "\n- ".join(self.available_actions),
            in_room=self.in_room,
            current_location=self.current_location,
        )

        messages = [
            SystemMessage(content=ADVENTURE_PLAN_SYSTEM_PROMPT),
            HumanMessage(content=plan_prompt),
        ]

        plan = safe_invoke(
            self.llm, messages, label=f"plan/{self.player_name}"
        )
        if plan is None:
            # Upstream model is unavailable -- continue with a no-op plan so
            # the game doesn't crash; choose_action() will fall back to a
            # safe default action when it sees no useful guidance.
            return plan_prompt, "(LLM unavailable - defaulting to safe action)"
        self.add_token_usage(plan.usage_metadata)
        return plan_prompt, plan.content.strip()

    def choose_action(self, plan: str) -> int:
        action_prompt = ADVENTURE_ACTION_USER_PROMPT.format(
            player_name=self.player_name,
            player_role=self.role,
            plan=plan,
            actions="- " + "\n- ".join(self.available_actions),
        )

        messages = [
            SystemMessage(content=ADVENTURE_ACTION_SYSTEM_PROMPT),
            HumanMessage(content=action_prompt),
        ]

        # Small local models (gemma3:1b etc.) often emit malformed actions and
        # the upstream engine can also be transiently unavailable (e.g. a
        # vLLM pod restarting on the InnKube hub). Retry up to 3 times for
        # *either* failure mode; if still invalid, fall back to a safe action
        # (waiting in the current location, or the first available action).
        last_text = ""
        for attempt in range(3):
            chosen_action = safe_invoke(
                self.response_llm,
                messages,
                attempts=1,
                label=f"action/{self.player_name}",
            )
            if chosen_action is None:
                continue
            self.add_token_usage(chosen_action.usage_metadata)
            last_text = chosen_action.content.strip()
            try:
                action_idx, action_text = check_action_valid(
                    self.available_actions, last_text, self.player_name
                )
                return action_prompt, action_idx, action_text
            except ValueError:
                print(
                    f"\033[33m[AdventureAgent] {self.player_name} bad output "
                    f"(attempt {attempt + 1}/3): {last_text!r}\033[0m"
                )

        fallback_idx = self._safe_action_fallback_index()
        fallback_text = self.available_actions[fallback_idx]
        print(
            f"\033[33m[AdventureAgent] {self.player_name} falling back to "
            f"{fallback_text!r}\033[0m"
        )
        return action_prompt, fallback_idx, fallback_text

    def _safe_action_fallback_index(self) -> int:
        for i, action in enumerate(self.available_actions):
            if "wait" in action.lower():
                return i
        return 0
