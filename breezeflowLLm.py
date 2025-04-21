"this is a Breezeflow LLM plugin for LiveKit, just a test code snippet, actual code is in breeze-voice/livekit/plugins/breezeflow/__init__.py"

import json
import time
import logging
from dataclasses import asdict, dataclass
from typing import Any, NotRequired, TypedDict

import aiohttp
from livekit.agents import llm
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, APIConnectOptions, NotGivenOr, NOT_GIVEN

logger = logging.getLogger(__name__)

@dataclass
class BreezeflowMessage:
    """Message format for Breezeflow API"""
    id: str
    text: str
    role: str

class BreezeflowConfig(TypedDict):
    chatbot_id: str
    api_url: NotRequired[str]

class BreezeflowLLM(llm.LLM):
    """Implementation of Breezeflow's chat API as an LLM plugin"""

    def __init__(
        self,
        *,
        chatbot_id: str,
        api_url: str = "https://breezeflow.io/api/agent/chat",
    ):
        super().__init__()
        self._chatbot_id = chatbot_id
        self._api_url = api_url

    async def chat(
        self,
        *,
        chat_ctx: llm.ChatContext,
        tools: list[llm.FunctionTool] | None = None,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
        **kwargs: Any,
    ) -> llm.LLMStream:
        return BreezeflowStream(
            llm=self,
            chat_ctx=chat_ctx,
            tools=tools or [],
            chatbot_id=self._chatbot_id,
            api_url=self._api_url,
            conn_options=conn_options,
        )

class BreezeflowStream(llm.LLMStream):
    def __init__(
        self,
        llm: BreezeflowLLM,
        *,
        chat_ctx: llm.ChatContext,
        tools: list[llm.FunctionTool],
        chatbot_id: str,
        api_url: str,
        conn_options: APIConnectOptions,
    ):
        super().__init__(llm, chat_ctx=chat_ctx, tools=tools, conn_options=conn_options)
        self._chatbot_id = chatbot_id
        self._api_url = api_url

    async def _run(self) -> None:
        try:
            # Convert context messages to Breezeflow format
            messages = []
            for msg in self._chat_ctx.messages:
                if msg.text:
                    messages.append(
                        BreezeflowMessage(
                            id=str(time.time()),
                            text=msg.text,
                            role="assistant" if msg.role == "model" else msg.role
                        )
                    )

            # Get the last user message
            last_message = messages[-1].text if messages else ""
            messages_dict = [asdict(msg) for msg in messages]

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self._api_url}?id={self._chatbot_id}",
                    headers={"Content-Type": "application/json"},
                    json={
                        "message": last_message,
                        "messages": messages_dict,
                    }
                ) as response:
                    if not response.ok:
                        raise llm.APIStatusError(
                            f"Breezeflow API error: {response.status}",
                            status_code=response.status
                        )

                    async for chunk in response.content:
                        chunk_text = chunk.decode('utf-8').strip()
                        if not chunk_text:
                            continue
                            
                        if chunk_text.startswith("0:"):
                            try:
                                content = json.loads(chunk_text[2:])
                                self._event_ch.send_nowait(
                                    llm.ChatChunk(
                                        id=str(time.time()),
                                        delta=llm.ChoiceDelta(
                                            content=content,
                                            role="assistant"
                                        )
                                    )
                                )
                            except json.JSONDecodeError as e:
                                logger.error(f"Error parsing chunk: {e}")

        except Exception as e:
            logger.exception("Error in Breezeflow service")
            raise llm.APIConnectionError(retryable=True) from e

# Example usage:
"""
llm=breezeflow.BreezeflowLLM(
    chatbot_id="your-chatbot-id",
)
"""