import os
import json
import asyncio
import logging
import aiohttp
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv(dotenv_path="../.env.local")

async def test_breezeflow_api():
    """Test direct communication with Breezeflow API"""
    chatbot_id = "b59bfa1b-695b-4033-9b49-e715ca3fd7f9"  # Your chatbot ID
    api_url = "https://obscure-engine-9ggw76grg563xrww-3000.app.github.dev/api/agent/chat"
    
    # Test conversation
    messages = [
        {
            "id": "1",
            "role": "system",
            "text": "You are a helpful assistant."
        },
        {
            "id": "2",
            "role": "user",
            "text": "What can you tell me about Python?"
        }
    ]

    try:
        async with aiohttp.ClientSession() as session:
            logger.info(f"Sending request to Breezeflow API...")
            async with session.post(
                f"{api_url}?id={chatbot_id}",
                headers={"Content-Type": "application/json"},
                json={
                    "message": messages[-1]["text"],
                    "messages": messages
                }
            ) as response:
                logger.info(f"Response status: {response.status}")
                
                if not response.ok:
                    error_text = await response.text()
                    logger.error(f"API error: {error_text}")
                    return

                accumulated_response = ""
                async for chunk in response.content:
                    chunk_text = chunk.decode('utf-8').strip()
                    if not chunk_text:
                        continue
                    
                    logger.debug(f"Raw chunk: {chunk_text}")
                    
                    if chunk_text.startswith("0:"):
                        try:
                            content = json.loads(chunk_text[2:])
                            logger.info(f"Received chunk: {content}")
                            accumulated_response += content
                        except json.JSONDecodeError as e:
                            logger.error(f"Error parsing chunk: {e}")

                logger.info(f"Complete response: {accumulated_response}")

    except Exception as e:
        logger.error(f"Error: {e}")
        raise

async def test_error_cases():
    """Test API error handling"""
    invalid_chatbot_id = "invalid-id"
    api_url = "https://breezeflow.io/api/agent/chat"
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{api_url}?id={invalid_chatbot_id}",
                headers={"Content-Type": "application/json"},
                json={
                    "message": "Hello",
                    "messages": [{"id": "1", "role": "user", "text": "Hello"}]
                }
            ) as response:
                logger.info(f"Error test status: {response.status}")
                error_text = await response.text()
                logger.info(f"Error response: {error_text}")
                
    except Exception as e:
        logger.error(f"Expected error: {e}")

if __name__ == "__main__":
    async def run_tests():
        logger.info("Testing Breezeflow API directly...")
        await test_breezeflow_api()
        logger.info("\nTesting error cases...")
        await test_error_cases()
    
    asyncio.run(run_tests())
