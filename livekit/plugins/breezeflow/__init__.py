from __future__ import annotations

import json
import time
import logging
import aiohttp
from typing import Any
from dataclasses import dataclass

from livekit.agents import (
    APIConnectionError, 
    APIStatusError,
    APITimeoutError,
    llm,
)
from livekit.agents.types import (
    NOT_GIVEN,
    APIConnectOptions,
    NotGivenOr,
    DEFAULT_API_CONNECT_OPTIONS
)

logger = logging.getLogger(__name__)

@dataclass
class _LLMOptions:
    chatbot_id: str
    api_url: str

class LLM(llm.LLM):
    def __init__(
        self,
        *,
        chatbot_id: str,
        api_url: str = "https://staging.breezeflow.io/api/agent/chat",
    ) -> None:
        super().__init__()
        self._opts = _LLMOptions(
            chatbot_id=chatbot_id,
            api_url=api_url,
        )

    def chat(
        self,
        *,
        chat_ctx: llm.ChatContext,
        tools: list | None = None,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
        extra_kwargs: NotGivenOr[dict[str, Any]] = NOT_GIVEN,
        fnc_ctx: Any = None,
    ) -> LLMStream:
        return LLMStream(
            self,
            chatbot_id=self._opts.chatbot_id,
            api_url=self._opts.api_url,
            chat_ctx=chat_ctx,
            conn_options=conn_options,
            fnc_ctx=fnc_ctx,
        )

class LLMStream(llm.LLMStream):
    def __init__(
        self,
        llm: LLM,
        *,
        chatbot_id: str,
        api_url: str,
        chat_ctx: llm.ChatContext,
        conn_options: APIConnectOptions,
        fnc_ctx: Any = None,
    ) -> None:
        super().__init__(
            llm=llm, 
            chat_ctx=chat_ctx, 
            conn_options=conn_options,
            fnc_ctx=fnc_ctx
        )
        self._chatbot_id = chatbot_id
        self._api_url = api_url

    async def _run(self) -> None:
        messages = []
        for msg in self._chat_ctx.messages:
            messages.append({
                "id": str(time.time()),
                "text": msg.content,
                "role": "assistant" if msg.role == "model" else msg.role
            })

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self._api_url}?id={self._chatbot_id}",
                    headers={"Content-Type": "application/json"},
                    json={
                        "message": messages[-1]["text"],
                        "messages": messages,
                    }
                ) as response:
                    if not response.ok:
                        raise APIStatusError(
                            f"Breezeflow API error: {response.status}",
                            status_code=response.status,
                            request_id="",
                            body="",
                            retryable=True
                        )

                    async for line in response.content:
                        line = line.decode().strip()
                        if line and line.startswith("0:"):
                            try:
                                content = json.loads(line[2:])
                                chunk = llm.ChatChunk(
                                    request_id=str(time.time()),
                                    choices=[llm.Choice(delta=llm.ChoiceDelta(role="assistant", content=content))]
                                )
                                self._event_ch.send_nowait(chunk)
                            except json.JSONDecodeError:
                                continue

        except aiohttp.ClientError as e:
            raise APITimeoutError(retryable=True) from e
        except Exception as e:
            raise APIConnectionError(retryable=True) from e
