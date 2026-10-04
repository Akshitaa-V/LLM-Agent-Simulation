from typing import Any

from langchain.schema import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import Field

from among_them.llm_factory import make_llm, safe_invoke
from among_them.llm_prompts import (
    DISCUSSION_RESPONSE_SYSTEM_PROMPT,
    DISCUSSION_RESPONSE_USER_PROMPT,
    DISCUSSION_SYSTEM_PROMPT,
    DISCUSSION_USER_PROMPT,
)
from among_them.rag.retriever import get_discussion_context

from .base_agent import Agent


class DiscussionAgent(Agent):
    llm: ChatOpenAI | None = None
    llm_model_name: str
    history: str = Field(default="")
    messages: str = Field(default_factory=list)

    def __init__(self, **data):
        super().__init__(**data)
        self.init_llm()

    def init_llm(self):
        self.llm = make_llm(self.llm_model_name, temperature=0.5)

    def act(
        self,
        observations: str,
        messages: str,
    ) -> Any:
        self.history = observations
        self.messages = messages

        points_prompt, points = self.create_discussion_points()
        response_prompt, response = self.respond_to_statements(points=points)
        self.responses.append(points)
        self.responses.append(response)
        return [points_prompt, response_prompt], response

    def create_discussion_points(self) -> str:
        rag_context = ""
        if self.injected_strategy:
            rag_context = (
                "--- Strategy for your role ---\n" + self.injected_strategy
            )
        elif self.use_rag:
            rag_context = get_discussion_context(
                role=self.role,
                recent_messages=(
                    self.messages[-300:] if self.messages else ""
                ),
            )

        system_message = SystemMessage(
            content=DISCUSSION_SYSTEM_PROMPT.format(your_name=self.player_name)
        )
        user_message = HumanMessage(
            content=DISCUSSION_USER_PROMPT.format(
                rag_context=rag_context,
                player_name=self.player_name,
                player_role=self.role,
                history=self.history,
                messages=self.messages,
            )
        )
        discussion_points = safe_invoke(
            self.llm,
            [system_message, user_message],
            label=f"discussion-points/{self.player_name}",
        )
        if discussion_points is None:
            return user_message.content, "(no discussion points - LLM unavailable)"
        self.add_token_usage(discussion_points.usage_metadata)
        return user_message.content, discussion_points.content.strip()

    def respond_to_statements(self, points: str) -> str:
        system_message = SystemMessage(
            content=DISCUSSION_RESPONSE_SYSTEM_PROMPT.format(your_name=self.player_name)
        )
        user_message = HumanMessage(
            content=DISCUSSION_RESPONSE_USER_PROMPT.format(
                player_name=self.player_name,
                player_role=self.role,
                points=points,
                messages=self.messages,
            )
        )
        response = safe_invoke(
            self.llm,
            [system_message, user_message],
            label=f"discussion-response/{self.player_name}",
        )
        if response is None:
            return user_message.content, "[no comment - LLM unavailable]"
        self.add_token_usage(response.usage_metadata)
        return user_message.content, response.content.strip()
