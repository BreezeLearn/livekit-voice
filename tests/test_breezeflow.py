import os
import asyncio
import logging
from dotenv import load_dotenv

from livekit.agents import llm
from livekit.plugins import breezeflow

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv(dotenv_path="../.env.local")

async def test_breezeflow_chat():
    """Test basic chat functionality with Breezeflow API"""
    
    # Initialize LLM with your chatbot ID
    llm_client = breezeflow.LLM(
        chatbot_id="b59bfa1b-695b-4033-9b49-e715ca3fd7f9",  # Replace with your chatbot ID
    )
    
    # Create a test conversation context
    chat_ctx = llm.ChatContext()
    chat_ctx.append(
        role="system",
        text="You are a helpful assistant."
    )
    chat_ctx.append(
        role="user",
        text="Hello, how are you?"
    )
    
    try:
        # Start chat stream
        logger.info("Starting chat stream...")
        stream = llm_client.chat(chat_ctx=chat_ctx)
        
        # Collect response
        response = []
        async for chunk in stream:
            if chunk.delta and chunk.delta.content:
                logger.info(f"Received chunk: {chunk.delta.content}")
                response.append(chunk.delta.content)
        
        full_response = "".join(response)
        logger.info(f"Full response: {full_response}")
        
        # Basic assertions
        assert full_response, "Response should not be empty"
        assert len(response) > 0, "Should receive at least one chunk"
        
    except Exception as e:
        logger.error(f"Test failed with error: {e}")
        raise

async def test_error_handling():
    """Test error handling with invalid chatbot ID"""
    
    # Initialize LLM with invalid chatbot ID
    llm_client = breezeflow.LLM(
        chatbot_id="invalid-id",
    )
    
    chat_ctx = llm.ChatContext()
    chat_ctx.append(
        role="user",
        text="Hello"
    )
    
    try:
        stream = llm_client.chat(chat_ctx=chat_ctx)
        async for _ in stream:
            pass
        assert False, "Should raise an error for invalid chatbot ID"
    except Exception as e:
        logger.info(f"Expected error received: {e}")
        assert True, "Should catch error for invalid chatbot ID"

if __name__ == "__main__":
    # Run tests
    async def run_tests():
        await test_breezeflow_chat()
        await test_error_handling()
    
    asyncio.run(run_tests())
