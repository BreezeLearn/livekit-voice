import requests
import logging
from openai import AzureOpenAI
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue


import os
load_dotenv()

logger = logging.getLogger(__name__)

# Default values for environment variables
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")

client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

azure_client = AzureOpenAI(
    api_key = os.getenv("AZURE_OPENAI_API_KEY", "your-api-key-here"),
    api_version = "2024-10-21",
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "https://breezeopenai.openai.azure.com/")
)


systemPromptTemplate = """
You're a friendly product guide for {company_name}—think helpful friend, not corporate chatbot. Your job: answer questions, simplify decisions, and help visitors find what they need.
How You Sound

Conversational and warm (like texting a knowledgeable friend)
Match their energy—casual with casual, professional with professional
Skip the jargon and sales-speak
Keep answers short (2-3 sentences max)
Your voice should be very very loud, energetic and inviting like a product guide, not a customer support agent.
What You Do

Answer questions using the company's knowledge base only
Explain features in plain English
Suggest next steps only when they ask or it feels natural
Say "I don't have enough info on that" when you're unsure
End with natural follow-ups that keep the conversation going

Sample Responses

Opening: "Hey! I'm here to help you find what you're looking for. What's on your mind?"
Confused: "Hmm, can you say that differently?"
Vague question: "Are you thinking about pricing, features, or something else?"
Don't know: "I don't have enough details on that—want me to connect you with the team?"

Key Rules

Never mention you're using a knowledge base
Only offer demos/CTAs when they directly ask
Sound human, not robotic
Be helpful, not pushy
do not entertain any questions outside of your core responsibilities as a product guide, this includes causual conversation, jokes, or any other topics that are not related to the company's products or services.

You must pull answer from the knowledgebase for whatever question you want to answer, never answer outside of the knowledgebase.

first message must be "Hey, I’m your AI guide—here to help you get answers fast, even the ones you might not find on the website. Ask me anything—I’d love to help you."
company_info = {company_info}
"""



def getAgentDetails(agent_id):
    is_staging = os.getenv("IS_STAGING", "false")
    url = f"https://app.breezeflow.ai/api/v1/agent?id={agent_id}"

    # Check if the environment is staging
    if is_staging.lower() == "true":
        url = f"https://staging.breezeflow.io/api/v1/agent?id={agent_id}"
    headers = {"Authorization": "Bearer " + os.getenv("BREEZE_API_KEY", "yto1ad8ckbk87xjunxrq7mqdpbv4id")}
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        data = response.json()

        if "data" in data:
            agent_data = data["data"]
            # company_info = f"{agent_data.get('description', 'No description provided')} — Your name: {agent_data.get('name', 'Unknown')}, Tone: {agent_data.get('tone', 'Not specified')}"
            company_info = f"{agent_data.get('description', 'No description provided')} — Your name: {agent_data.get('name', 'Unknown')}, Tone: {agent_data.get('tone', 'Not specified')}"
            company = agent_data.get("company", "Unknown")
            company_name = company.get("company_name", "Unknown")
            
            system_prompt = systemPromptTemplate.format(company_info=company_info, company_name=company_name)
            logger.info(f"Retrieved system prompt for agent {agent_id}: {system_prompt}")
            return system_prompt
        else:
            raise ValueError("Invalid response format")
    except Exception as e:
        return f"Failed to retrieve agent details: {str(e)}"

# Example usage:
# agent_id = "b59bfa1b-695b-4033-9b49-e715ca3fd7f9"
# print(getAgentDetails(agent_id))


def getCollectionName(agent_id):
    is_staging = os.getenv("IS_STAGING", "false")
    url = f"https://app.breezeflow.ai/api/v1/agent?id={agent_id}"

    # Check if the environment is staging
    if is_staging.lower() == "true":
        url = f"https://staging.breezeflow.io/api/v1/agent?id={agent_id}"
    headers = {"Authorization": "Bearer " + os.getenv("BREEZE_API_KEY", "yto1ad8ckbk87xjunxrq7mqdpbv4id")}
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        data = response.json()

        if "data" in data and "KnowledgeBase" in data["data"]:
            agent_data = data["data"]
            company = agent_data.get("company", "Unknown")
            companyId = company.get("_id", "Unknown")
            knowledge_base = data["data"]["KnowledgeBase"]
            
            if knowledge_base and len(knowledge_base) > 0:
                return knowledge_base[0].get("collectionName"), companyId
        return None
    except Exception as e:
        logger.error(f"Failed to retrieve collection name: {str(e)}")
        return None


def getEmbedding(text):
    response = azure_client.embeddings.create(
        input = text,
        model= "text-embedding-3-large"
    )
    # Extract the embedding vector from the response
    return response.data[0].embedding


def queryQdrant(query, collection_name, companyId):
    logger.info(f"Querying Qdrant with collection name: {collection_name} - companyId: {companyId}")
    query_embedding = getEmbedding(query)
    response = client.query_points(
        collection_name=collection_name,
        query=query_embedding,
        limit=5,
        with_payload=True,
        query_filter=Filter(
        must=[FieldCondition(key="companyId", match=MatchValue(value=companyId))]
    ),
    )
    return response