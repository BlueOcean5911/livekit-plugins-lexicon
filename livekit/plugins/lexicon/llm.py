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

class LLMConfig(BaseModel):
    base_url: str
    agent_id: str
    user_id: str
    chat_id: str
    chatbot_id: str

class LLM(llm.LLM):
    def __init__(
        self,
        *,
        config: LLMConfig,
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
        config: LLMConfig,
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
                f"{self.config.base_url}?message={user_msg.content}",
            )  
            
            if response.status_code != 200:
                raise APIConnectionError()
            
            message = response.json()["message"]
            
            self._event_ch.send_nowait(
                llm.ChatChunk(
                    request_id="",
                    choices=[
                        llm.Choice(
                            delta=llm.ChoiceDelta(
                                role="assistant",
                                content=message,
                            )
                        )
                    ],
                )
            )
            pass
        except Exception as e:
            raise APIConnectionError() from e
