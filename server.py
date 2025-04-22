# server.py
import logging
import os
# Remove subprocess, sys imports
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

import livekit.api as livekit_api

# Load environment variables from .env.local
# These will be inherited by the subprocesses
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("agent-dispatcher-server")

# --- App State --- 
class AppState:
    def __init__(self):
        self.livekit_client: livekit_api.LiveKitAPI | None = None

app_state = AppState()
# -----------------

# --- FastAPI App ---
app = FastAPI()

@app.on_event("startup")
async def startup_event():
    """
    Initialize LiveKit API client on server startup.
    """
    logger.info("Server starting up...")
    try:
        # Initialize LiveKit API client
        livekit_url = os.environ.get("LIVEKIT_URL")
        # Ensure the URL for the API client doesn't have the ws(s) protocol prefix
        if livekit_url and livekit_url.startswith("ws"):
            api_url = livekit_url.replace("ws", "http", 1)
        else:
            api_url = livekit_url # Assume http/https if not ws/wss

        livekit_api_key = os.environ.get("LIVEKIT_API_KEY")
        livekit_api_secret = os.environ.get("LIVEKIT_API_SECRET")
        if not all([api_url, livekit_api_key, livekit_api_secret]):
             raise ValueError("LIVEKIT_URL (formatted for HTTP/S), LIVEKIT_API_KEY, and LIVEKIT_API_SECRET must be set in .env.local")
        
        # Use the http/https URL for the API client
        app_state.livekit_client = livekit_api.LiveKitAPI(api_url, livekit_api_key, livekit_api_secret)
        # Optional: Test connection or list rooms to verify client works
        # await app_state.livekit_client.room.list_rooms()
        logger.info("LiveKit API client initialized.")

    except Exception as e:
        logger.error(f"Error during startup initializing LiveKitAPI: {e}", exc_info=True)
        # We might want to prevent startup if the API client fails
        # For now, log the error; endpoints will fail if client is None
    logger.info("Server startup complete.")

@app.on_event("shutdown")
async def shutdown_event():
    """
    Cleanly close the LiveKit API client on server shutdown.
    """
    logger.info("Server shutting down...")
    if app_state.livekit_client:
        await app_state.livekit_client.aclose()
        logger.info("LiveKit API client closed.")
    logger.info("Server shutdown complete.")

# --- Request Models ---
class StartAgentRequest(BaseModel):
    room_name: str
    participant_identity: str | None = None
    agent_id: str # Passed as metadata
    agent_name: str | None = None # Optional: target specific agent worker by name

# --- Remove Agent Process Management Section --- 
# running_agents = {}
# -----------------------------------------------

@app.post("/start-agent")
async def start_agent_endpoint(request: StartAgentRequest):
    """
    API endpoint to dispatch an agent job for a specific room.
    Does NOT launch agent processes anymore.
    """
    agent_id_metadata = request.agent_id
    room_name = request.room_name
    target_agent_name = request.agent_name
    
    logger.info(f"Received request to dispatch agent job for room: {room_name} with agent_id (metadata): {agent_id_metadata}" + 
                (f" targeting agent name: {target_agent_name}" if target_agent_name else ""))

    if not app_state.livekit_client:
         logger.error("LiveKit API client not initialized. Cannot create dispatch.")
         raise HTTPException(status_code=503, detail="LiveKit API client not ready.")

    # --- Remove Launch Agent Process Section --- 
    # ... (subprocess code removed) ...
    # -------------------------------------------

    # --- Create Agent Dispatch --- 
    try:
        logger.info(f"Creating agent dispatch for room: {room_name} with agent_id: {agent_id_metadata} in metadata")
        dispatch_request = livekit_api.CreateAgentDispatchRequest(
            room=agent_id_metadata,
            metadata=f"{{\"agent_id\": \"{agent_id_metadata}\"}}", # Pass agent_id as metadata
            agent_name=target_agent_name # Pass optional agent name target
        )
        dispatch = await app_state.livekit_client.agent_dispatch.create_dispatch(dispatch_request)
        logger.info(f"Created agent dispatch successfully: ID {dispatch.id}")
        
        # Return success response
        return {
            "message": f"Agent dispatch created for room {room_name}.",
            "dispatch_id": dispatch.id
        }

    except Exception as e:
        logger.error(f"Failed to create agent dispatch for room {room_name}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to create agent dispatch: {e}")
    # -----------------------------

# Optional: Endpoint to check status or stop agents could be added here.

# # --- Main Execution (for running with uvicorn) ---
# if __name__ == "__main__":
#     import uvicorn
#     # Ensure LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET are in .env.local
#     uvicorn.run(app, host="0.0.0.0", port=8080) 