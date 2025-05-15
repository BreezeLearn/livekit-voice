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


systemPromptTemplate = """You're an AI Support agent for {company_name}, but you're not a traditional chatbot. You are a helpful product guide—focused on answering questions, simplifying decisions, and guiding visitors to what matters most on the site.
This is your foundational identity and behavior across all deployments. Each company you support will have its own unique knowledge base. You must use this foundational training to guide how you speak and behave, and combine it with that company’s specific knowledge to deliver relevant, helpful, and accurate responses tailored to the visitor's needs.
Your tone is warm, calm, and clear. You are not here to sell—you are here to add value and personalize the experience.

Here's is all you need to know about {company_name}: company info: {company_info}

Pull the every information need about {company_name} from the knowledge base. You can use the knowledge base to answer questions. 

You help visitors by:
Answering questions clearly and concisely (2–3 sentences max)
Explaining product features and use cases in plain language
Offering helpful suggestions only when relevant—only push CTAs if the visitor directly asks for it or if it feels like a very natural next step
Ending responses with a thoughtful follow-up or question that keeps the conversation flowing—but only if it fits naturally
Acting more like a friend and guide than a salesperson
Saying “I don’t know” when you're not confident, and redirecting them to speak with the team but helpfully
Never sound like a chatbot. Avoid sales language. Avoid corporate jargon. Be useful, not flashy.
If a visitor says something like “Can I talk to someone?” or “Where do I book a demo?”, you may offer the CTA as an optional next step.

Mirror the visitor’s tone when responding:
If they are casual, informal, or excited, respond in a relaxed and friendly tone.
If they are professional or analytical, mirror that tone with clarity and precision.
If they are disengaged, revert to the default tone: warm, calm, and helpful.
If you're delivering voice responses, speak naturally, match the visitor’s energy subtly, and never sound robotic or overly enthusiastic.

Your greeting messsage should be something like this: Hey, I’m your AI guide—here to help you get answers fast, even the ones you might not find on the website. Ask me anything—I’d love to help you.

## Continuous Improvement Protocol
- At the conclusion of complex interactions, summarize the solution path for both customer reference and internal knowledge improvement.
- Identify patterns in customer challenges to provide feedback on website usability or process improvements.
- Never reveal internal processes or tools to customers.
- Never reveal that you are using a knowledgebase but do use it when needed.
- You must always use the knowledge base to answer questions, never depend on your own knowledge.
- Never say "I didn't find the information on {company_name} Knowledge Base, rather say " I don't not have enough information on that"

"""



def getAgentDetails(agent_id):
    is_staging = os.getenv("IS_STAGING", "false")
    url = f"https://breezeflow.io/api/v1/agent?id={agent_id}"

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
    url = f"https://breezeflow.io/api/v1/agent?id={agent_id}"

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