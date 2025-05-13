import logging
import asyncio
import json
from livekit import rtc, api
from livekit.agents import (
    Agent,
    AgentSession,
    RoomInputOptions,
    RoomOutputOptions,
    get_job_context,
    JobContext,
    WorkerOptions,
    cli,
    function_tool,
    ToolError,
    RunContext
)
from livekit.agents.llm import ImageContent, ChatContext, ChatMessage
from livekit.api import RoomParticipantIdentity

from dotenv import load_dotenv
from livekit.plugins import (
    openai,
    noise_cancellation,
    silero
)
from livekit.plugins.turn_detector.multilingual import MultilingualModel
from openai.types.beta.realtime.session import TurnDetection
from prompt import getAgentDetails, queryQdrant, getCollectionName

load_dotenv()
logger = logging.getLogger("voice-agent")


class Assistant(Agent):
    def __init__(self, instructions=str) -> None:
        self._latest_frame = None
        self._video_stream = None
        self._tasks = []
        super().__init__(instructions=instructions)


    
    @function_tool()
    async def lookup_knowledgebase(
        context: RunContext,
        query: str,
    ) -> dict:
        """Look information in the knowledge base of the company you're representing. Use this to answer users questions you're not sure about."""

        room = get_job_context().room
        participant_identity = next(iter(room.remote_participants))
        async with api.LiveKitAPI() as lkapi:
            res = await lkapi.room.get_participant(RoomParticipantIdentity(
                room=room.name,
                identity=participant_identity,
            ))

            collection_name, companyId = getCollectionName(res.name)

            if not collection_name:
                raise ToolError("Knowledge base not found. Please try again later.")
            response = queryQdrant(query, collection_name, companyId)
            if not response or not response.points:
                raise ToolError("No results found in the knowledge base.")

            # Process and format the response
            results = []
            for point in response.points:
                if hasattr(point, 'payload') and point.payload and 'content' in point.payload:
                    results.append({
                        'content': point.payload['content'],
                        'score': point.score
                    })
            
            if not results:
                raise ToolError("No content found in the knowledge base.")

            return {'results': results}

    # @function_tool()
    # async def label_page_elements(
    #     context: RunContext,
    #     label: str,
    #     action: str,
    # ):
    #     """Label page elements using Javascript.
    #     action: "label" or "click" or "scroll"
    #     if action is "scroll", label is the scroll direction (up or down) 
    #     if action is "label", label is 0
    #     if action is "click", label is the numerical label of the page elements to be clicked

    #     This function assigns numerical labels page elements.
    #     Returns:
    #         A dictionary with the labels of the page elements
    #     """
    #     try:
    #         room = get_job_context().room
    #         participant_identity = next(iter(room.remote_participants))
    #         logger.info(f"Participant identity: {participant_identity}")
    #         response = await room.local_participant.perform_rpc(
    #             destination_identity=participant_identity,
    #             method="labelPageElements",
    #             payload=json.dumps({
    #                 "labeled": True,
    #                 "label": label,
    #                 "action": action,
    #             }),
    #         )
    #         return response
    #     except Exception:
    #         raise ToolError("Unable to label page elements. Please try again later.")
    
    async def on_enter(self):
        room = get_job_context().room

        # Find the first video track (if any) from the remote participant
        remote_participant = list(room.remote_participants.values())[0]
        video_tracks = [publication.track for publication in list(remote_participant.track_publications.values()) if publication.track.kind == rtc.TrackKind.KIND_VIDEO]
        if video_tracks:
            self._create_video_stream(video_tracks[0])
        
        # Watch for new video tracks not yet published
        @room.on("track_subscribed")
        def on_track_subscribed(track: rtc.Track, publication: rtc.RemoteTrackPublication, participant: rtc.RemoteParticipant):
            if track.kind == rtc.TrackKind.KIND_VIDEO:
                self._create_video_stream(track)
                        
    async def on_user_turn_completed(self, turn_ctx: ChatContext, new_message: ChatMessage) -> None:
        # Add the latest video frame, if any, to the new message
        if self._latest_frame:
            new_message.content.append(ImageContent(image=self._latest_frame))
            self._latest_frame = None
    
    # Helper method to buffer the latest video frame from the user's track
    def _create_video_stream(self, track: rtc.Track):
        # Close any existing stream (we only want one at a time)
        if self._video_stream is not None:
            self._video_stream.close()

        # Create a new stream to receive frames    
        self._video_stream = rtc.VideoStream(track)
        async def read_stream():
            async for event in self._video_stream:
                # Store the latest frame for use later
                self._latest_frame = event.frame
        
        # Store the async task
        task = asyncio.create_task(read_stream())
        task.add_done_callback(lambda t: self._tasks.remove(t))
        self._tasks.append(task)


async def entrypoint(ctx: JobContext):
    await ctx.connect()

    participant = await ctx.wait_for_participant()
    logger.info(f"starting voice assistant for participant {participant.identity}")

    async with api.LiveKitAPI() as lkapi:
        res = await lkapi.room.get_participant(RoomParticipantIdentity(
        room=ctx.job.room.name,
        identity=participant.identity,
        ))
        logger.info(f"Participant info: {res.identity}, {res.name}, {res.metadata}")

    systemPrompt = getAgentDetails(participant.name)

#     session = AgentSession(
#         stt=openai.STT(
#             model="gpt-4o-transcribe",
#         ),
#         llm=openai.LLM(model="gpt-4o"),
#         tts=openai.TTS(
#         model="gpt-4o-mini-tts",
#         voice="alloy",
#         instructions="""Affect/personality: A cheerful guide 

# Tone: Friendly, clear, and reassuring, creating a calm atmosphere and making the listener feel confident and comfortable and be more energetic , enthusiatic.

# Pronunciation: Clear, articulate, and steady, ensuring each instruction is easily understood while maintaining a natural, conversational flow.

# Pause: Brief, purposeful pauses after key instructions (e.g., "cross the street" and "turn right") to allow time for the listener to process the information and follow along.

# Emotion: Warm and supportive, conveying empathy and care, ensuring the listener feels guided and safe throughout the journey."""
# ,
#     ),
#         vad=silero.VAD.load(),
#         turn_detection=MultilingualModel(),
#     )

    # logger.info(f"instructions: {systemPrompt}")
    session = AgentSession(
        llm=openai.realtime.RealtimeModel(
            voice="alloy",
            turn_detection=TurnDetection(
                type="semantic_vad",
                eagerness="auto",
                create_response=True,
                interrupt_response=True,
            ),
        )
    ) 
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
        instructions="say: Hey, I’m your AI guide—here to help you get answers fast, even the ones you might not find on the website. Ask me anything—I’d love to help you.",
    )


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
