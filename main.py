from dotenv import load_dotenv

from livekit import agents
import json
from livekit.agents import ToolError
from livekit.agents import (
    AgentSession,
    function_tool,
    Agent,
    RunContext,
    RoomInputOptions,
    RoomOutputOptions,
    get_job_context
)
from livekit.plugins import (
    openai,
    noise_cancellation,
    google,
)
from openai.types.beta.realtime.session import TurnDetection
from prompt import getAgentDetails
import logging
from livekit import api
from livekit.api import RoomParticipantIdentity


load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Assistant(Agent):
    def __init__(self, instructions=str) -> None:
        super().__init__(instructions=instructions)
    
    @function_tool()
    async def lookup_user(
        context: RunContext,
        user_id: str,
    ) -> dict:
        """Look up a user's information by ID."""
        return {"name": "John Doe", "email": "john.doe@example.com"}
    

    @function_tool()
    async def label_page_elements(
        context: RunContext,
        label: str,
        action: str,
    ):
        """Label page elements using Javascript.
        action: "label" or "click" or "scroll"
        if action is "scroll", label is the scroll direction (up or down) 
        if action is "label", label is 0
        if action is "click", label is the numerical label of the page elements to be clicked

        This function assigns numerical labels page elements.
        Returns:
            A dictionary with the labels of the page elements
        """
        try:
            room = get_job_context().room
            participant_identity = next(iter(room.remote_participants))
            logger.info(f"Participant identity: {participant_identity}")
            response = await room.local_participant.perform_rpc(
                destination_identity=participant_identity,
                method="labelPageElements",
                payload=json.dumps({
                    "labeled": True,
                    "label": label,
                    "action": action,
                }),
                # response_timeout=10.0 if high_accuracy else 5.0,
            )
            return response
        except Exception:
            raise ToolError("Unable to label page elements. Please try again later.")
        
    # @function_tool()
    # async def click_page_elements(
    #     context: RunContext,
    #     label: int,
    # ):
    #     """Click page elements using Javascript.
        
    #     This function sends a request to the remote participant to click page elements with a specific label (number).
    #     Returns:
    #         A dictionary with the labels of the page elements
    #     """
    #     try:
    #         room = get_job_context().room
    #         participant_identity = next(iter(room.remote_participants))
    #         logger.info(f"Participant identity: {participant_identity}, Label: {label}")
    #         response = await room.local_participant.perform_rpc(
    #             destination_identity=participant_identity,
    #             method="clickPageElements",
    #             payload=json.dumps({
    #                 "clicked": True,
    #                 "label": label
    #             })
    #             # response_timeout=10.0 if label else 5.0,
    #         )
    #         logger.info(f"Response: {response}")
    #         return response
    #     except Exception:
    #         raise ToolError("Unable to click page elements. Please try again later.")


async def entrypoint(ctx: agents.JobContext):
    await ctx.connect()

    
    # agent_id = ctx.job
    # logger.info(f"Agent ID: {agent_id}")
    participant = await ctx.wait_for_participant()
    logger.info(f"Starting voice assistant for participant {participant}")

    async with api.LiveKitAPI() as lkapi:
        res = await lkapi.room.get_participant(RoomParticipantIdentity(
        room=ctx.job.room.name,
        identity=participant.identity,
        ))
        logger.info(f"Participant info: {res.identity}, {res.name}, {res.metadata}")

    systemPrompt = getAgentDetails(participant.name)
    # logger.info(f"System prompt: {systemPrompt}")
    # Initialize the agent session with the Google Gemini model

    session = AgentSession(
        llm=google.beta.realtime.RealtimeModel(
            model="gemini-2.0-flash-exp",
            voice="Puck",
            temperature=0.6,
            instructions=systemPrompt,
        ),
    )

    # session = AgentSession(
    #     llm=openai.realtime.RealtimeModel(
    #         voice="coral",
    #         turn_detection=TurnDetection(
    #             type="semantic_vad",
    #             eagerness="auto",
    #             create_response=True,
    #             interrupt_response=True,
    #         ),
    #     )
    # )

    await session.start(
        room=ctx.room,
        agent=Assistant(instructions=systemPrompt),
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
            video_enabled=True,
            text_enabled=True,
            audio_enabled=True,
        ),
        room_output_options=RoomOutputOptions(transcription_enabled=True),
    )

    await session.generate_reply(
        instructions="Greet the user and enlighten them about yourself and the company.",
    )


if __name__ == "__main__":
    agents.cli.run_app(agents.WorkerOptions(entrypoint_fnc=entrypoint))
