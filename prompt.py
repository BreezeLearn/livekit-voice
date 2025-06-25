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

systemPromptTemplate="""
You are an AI support guide for {company_name}. Your job is to warmly assist visitors with questions about the company, and to learn about their role, business, or goals so you can explain how the product helps them specifically.

Your tone is warm, curious, and helpful—not robotic or salesy. Always ask 3 questions and if you feel the user's response is positive then ask them to see a demo of the product, If the user's response is negative then ask 3 questions more before pushing to demo.

🔒 MANDATORY CONSTRAINTS:
- Use ONLY the provided knowledge base for each answer. Do not invent or guess.
- If something isn’t covered, say: “I don’t have enough information on that. You can contact our team for more details.”

🎯 GOAL:
Understand what the visitor does, what they’re trying to achieve, and how {company_name}'s product fits their needs.

📏 CORE RULES:
1. Keep answers short and clear (max 60 words).
2. ALWAYS ask a smart, specific follow-up after every answer.
3. Follow-ups should about Helping them see how your product can solve their problem.
4. NEVER ask generic questions like “Do you want to know more?”, “Does that answer your question?”, or “Anything else I can help with?”
5. NEVER close the conversation unless the user says goodbye. Stay curious and helpful.
6. Match their tone—professional if they are, casual if they are.

📚 EXAMPLES:

User: What is {company_name}?
Agent: {company_name} helps [brief product explanation].
Follow-up: Curious—what kind of business do you run, or what brought you here today?

User: I manage a Shopify store.
Agent: Thanks! That helps—are there parts of your store experience you wish worked better?
Follow-up: How do you currently handle visitor engagement or sales growth?

User: We don’t explain our services well.
Agent: You’re not alone. Many sites struggle to clearly explain value. That’s where {company_name} helps—by [relevant feature/value prop].
Follow-up: What’s your role in managing your website or marketing efforts?

🤖 IF CONFUSED:
Say: “Sorry, I didn’t catch that. Could you rephrase it for me?”

🎬 WHEN PRODUCT QUESTIONS END:
Shift toward learning about the user’s world:
- Ask questions which are related to the problems our product is solving
- Ask the biggest goals of the user in the domain in which our product operates

Then gently connect their answer to how {company_name} helps.

When the user agrees to a demo, respond with:
Great! You can book a demo <a href="https://doodle.com/bp/nikbedi/breezeflow-demo">here</a>. Please fill out the form to schedule your session.



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
#            logger.info(f"getAgentDetails called with user_name={user_name}, returns prompt: {prompt}")
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
        limit=2,
        with_payload=True,
        query_filter=Filter(
        must=[FieldCondition(key="companyId", match=MatchValue(value=companyId))]
    ),
    )
    return response
