from __future__ import annotations

from typing import Literal, Union
from pydantic import BaseModel

from livekit.agents import (
    APIConnectionError,
    llm,
)
from livekit.agents.llm import ToolChoice
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, APIConnectOptions

import requests

from .log import logger

class ModelConfig(BaseModel):
    base_url: str
    agent_id: str
    user_id: str
    chat_id: str
    chatbot_id: str

class LLM(llm.LLM):
    def __init__(
        self,
        *,
        config: ModelConfig,
    ) -> None:
        super().__init__()
        self.config = config

    def chat(
        self,
        *,
        chat_ctx: llm.ChatContext,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
        fnc_ctx: llm.FunctionContext | None = None,
        temperature: float | None = None,
        n: int | None = 1,
        parallel_tool_calls: bool | None = None,
        tool_choice: Union[ToolChoice, Literal["auto", "required", "none"]]
        | None = None,
    ) -> "LLMStream":
        if fnc_ctx is not None:
            logger.warning("fnc_ctx is currently not supported with llama_index.LLM")

        return LLMStream(
            self,
            config=self.config,
            chat_ctx=chat_ctx,
            conn_options=conn_options,
        )


class LLMStream(llm.LLMStream):
    def __init__(
        self,
        llm: LLM,
        *,
        config: ModelConfig,
        chat_ctx: llm.ChatContext,
        conn_options: APIConnectOptions,
    ) -> None:
        super().__init__(
            llm, chat_ctx=chat_ctx, fnc_ctx=None, conn_options=conn_options
        )
        self.config = config

    async def _run(self) -> None:
        chat_ctx = self._chat_ctx.copy()
        user_msg = chat_ctx.messages.pop()

        if user_msg.role != "user":
            raise ValueError(
                "The last message in the chat context must be from the user"
            )

        assert isinstance(user_msg.content, str), (
            "user message content must be a string"
        )

        try:
            response =  requests.post(
                f"{self.config.base_url}/chat",
                data={
                    "agent_id": self.config.agent_id,
                    "user_id": self.config.user_id,
                    "chat_id": self.config.chat_id,
                    "chatbot_id": self.config.chatbot_id,
                    "message": user_msg.content,
                },
            )
            
            self._event_ch.send_nowait(
                llm.ChatChunk(
                    request_id="",
                    choices=[
                        llm.Choice(
                            delta=llm.ChoiceDelta(
                                role="assistant",
                                content=delta,
                            )
                        )
                    ],
                )
            )
            pass
        except Exception as e:
            raise APIConnectionError() from e
