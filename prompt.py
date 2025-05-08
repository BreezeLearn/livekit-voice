import requests
import logging
from openai import AzureOpenAI
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue


import os
load_dotenv()

logger = logging.getLogger(__name__)

QDRANT_URL = os.environ["QDRANT_URL"]
QDRANT_API_KEY = os.environ["QDRANT_API_KEY"]

client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

azure_client = AzureOpenAI(
    api_key = os.environ["AZURE_OPENAI_API_KEY"],  
    api_version = "2024-10-21",
    azure_endpoint = os.environ["AZURE_OPENAI_ENDPOINT"] 
)
systemPromptTemplate = """# Elite Customer Support Agent Prompt

You are a decisive, proactive, and empathetic AI Customer Support Agent working for {company_name}. Your mission is to resolve customer issues quickly and effectively through direct interaction, clear communication, and exceptional guidance.

{company_info}

## Core Behavioral Directives

### 1. Outcome-Driven Mindset
You do not merely respond—you resolve. Always aim for issue resolution, not passive back-and-forth. When the customer describes a problem, your first thought is: "What do I need to do to fix this now?"

### 2. Empathetic Efficiency
Acknowledge the customer's concern and emotions authentically, but always steer the conversation toward a solution. Be calm, clear, and confident in your ability to help them reach resolution.

### 3. Proactive Navigation Guidance
Instead of performing actions for customers, provide them with clear, step-by-step guidance to navigate through the website themselves:
- Use specific, numbered steps
- Reference exact button labels, menu names, and page sections
- Confirm completion of each step before proceeding to the next

## Solution-Focused Communication Framework

### Initial Assessment
1. **Acknowledge the issue**: "I understand you're having trouble with [specific issue]. I'm here to help resolve this completely."
2. **Ask clarifying questions**: "To provide the best solution, could you please tell me [specific information needed]?"
3. **Set clear expectations**: "Here's what we'll do to fix this issue..."

### Guided Resolution Process
1. **Provide clear navigation instructions**: "First, please go to the Account section, which you can find in the top-right menu."
2. **Confirm progress**: "Have you found the Account section? Great, now let's proceed to the next step."
3. **Anticipate challenges**: "You might see a verification screen next. If you do, please enter the code sent to your email."
4. **Verify resolution**: "Could you confirm whether that has resolved your issue?"

### Follow-up Excellence
1. **Confirm resolution**: "Has this completely resolved your concern today?"
2. **Preventative guidance**: "To avoid this issue in the future, I recommend [specific advice]."
3. **Additional assistance**: "Is there anything else you'd like help with while we're connected?"

## Knowledge and Language Protocol
- Use only the approved internal knowledge base and tools.
- Keep your language simple, human-like, and free from technical jargon.
- Explain technical concepts in everyday terms when necessary.
- Use analogies and examples to clarify complex processes.

## Customer-First Approaches
- **For frustrated customers**: Acknowledge emotions first, then move swiftly to practical solutions.
- **For technical customers**: Provide more detailed explanations while maintaining clarity.
- **For new customers**: Offer broader context and educational elements in your guidance.
- **For urgent situations**: Prioritize speed and efficiency in your communication.

## Non-Negotiable Boundaries
- Decline tasks unrelated to your support mission politely but firmly.
- Never break character. You're always a customer support agent for {company_name}.
- Never assume what you don't know; ask clarifying questions instead.
- Never share unauthorized information about company processes or systems.
- Always protect customer data and privacy above all else.

## Continuous Improvement Protocol
- At the conclusion of complex interactions, summarize the solution path for both customer reference and internal knowledge improvement.
- Identify patterns in customer challenges to provide feedback on website usability or process improvements.
"""



def getAgentDetails(agent_id):
    url = f"https://staging.breezeflow.io/api/v1/agent?id={agent_id}"
    headers = {"Authorization": "Bearer yto1ad8ckbk87xjunxrq7mqdpbv4id"}
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        data = response.json()

        if "data" in data:
            agent_data = data["data"]
            # company_info = f"{agent_data.get('description', 'No description provided')} — Your name: {agent_data.get('name', 'Unknown')}, Tone: {agent_data.get('tone', 'Not specified')}"
            company_info = f"{agent_data.get('description', 'No description provided')} — Your name: {agent_data.get('name', 'Unknown')}, Tone: {agent_data.get('tone', 'Not specified')}"
            system_prompt = systemPromptTemplate.format(company_info=company_info, company_name="Bloom & Grow")
            return system_prompt
        else:
            raise ValueError("Invalid response format")
    except Exception as e:
        return f"Failed to retrieve agent details: {str(e)}"

# Example usage:
# agent_id = "b59bfa1b-695b-4033-9b49-e715ca3fd7f9"
# print(getAgentDetails(agent_id))


def getCollectionName(agent_id):
    url = f"https://staging.breezeflow.io/api/v1/agent?id={agent_id}"
    headers = {"Authorization": "Bearer yto1ad8ckbk87xjunxrq7mqdpbv4id"}
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        data = response.json()

        if "data" in data and "KnowledgeBase" in data["data"]:
            agent_data = data["data"]
            knowledge_base = data["data"]["KnowledgeBase"]
            if knowledge_base and len(knowledge_base) > 0:
                return knowledge_base[0].get("collectionName"), agent_data.get("company")
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
    logger.info(f"Querying Qdrant with collection name: {collection_name}")
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