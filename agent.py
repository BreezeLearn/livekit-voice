import logging

from dotenv import load_dotenv
from livekit.agents import (
    AutoSubscribe,
    JobContext,
    JobProcess,
    WorkerOptions,
    cli,
    llm,
    metrics,
)
from livekit.agents.pipeline import VoicePipelineAgent
from livekit.plugins import (
    deepgram,
    noise_cancellation,
    silero,
    turn_detector
)
from breezeflowLLm import LLM as BreezeflowLLM

from livekit.plugins.deepgram import tts

load_dotenv()
logger = logging.getLogger("voice-agent")


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    # --- Get agent_id from job metadata --- 
    agent_id = ctx.job.metadata
    # if not agent_id:
    #     logger.error(f"Job metadata (agent_id) is missing or empty. Cannot configure LLM. metadata: {ctx.job.metadata}")
    #     # Decide how to handle this - maybe raise an error or use a default?
    #     # For now, let's raise to make the problem visible.
    #     raise ValueError("Agent ID not found in job metadata")
    # logger.info(f"Received agent_id from job metadata: {agent_id}")
    # --------------------------------------
    

    initial_ctx = llm.ChatContext().append(
        role="system",
        text=(
            "You are a voice assistant created by LiveKit. Your interface with users will be voice. "
            "You should use short and concise responses, and avoiding usage of unpronouncable punctuation. "
            "You were created as a demo to showcase the capabilities of LiveKit's agents framework."
        ),
    )

    logger.info(f"connecting to room {ctx.room.name}")
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    # Wait for the first participant to connect
    participant = await ctx.wait_for_participant()
    logger.info(f"starting voice assistant for participant {participant.identity}")

    # This project is configured to use Deepgram STT, OpenAI LLM and Cartesia TTS plugins
    # Other great providers exist like Cerebras, ElevenLabs, Groq, Play.ht, Rime, and more
    # Learn more and pick the best one for your app:
    # https://docs.livekit.io/agents/plugins

    logger.info(f"Starting voice agent for job 77 with agent_id {ctx.agent.name}")
    agent = VoicePipelineAgent(
        vad=ctx.proc.userdata["vad"],
        stt=deepgram.STT(),
        llm=BreezeflowLLM(
            chatbot_id=participant.identity,
            # temperature=0.8,
        ),
        tts=tts.TTS(
            model="aura-asteria-en",
        ),
        # use LiveKit's transformer-based turn detector
        turn_detector=turn_detector.EOUModel(),
        # minimum delay for endpointing, used when turn detector believes the user is done with their turn
        min_endpointing_delay=0.5,
        # maximum delay for endpointing, used when turn detector does not believe the user is done with their turn
        max_endpointing_delay=5.0,
        # enable background voice & noise cancellation, powered by Krisp
        # included at no additional cost with LiveKit Cloud
        noise_cancellation=noise_cancellation.BVC(),
        chat_ctx=initial_ctx,
    )

    usage_collector = metrics.UsageCollector()

    @agent.on("metrics_collected")
    def on_metrics_collected(agent_metrics: metrics.AgentMetrics):
        metrics.log_metrics(agent_metrics)
        usage_collector.collect(agent_metrics)

    agent.start(ctx.room, participant)

    # The agent should be polite and greet the user when it joins :)
    await agent.say("Hey, how can I help you today?", allow_interruptions=True)


if __name__ == "__main__":
    # No custom argument parsing or sys.argv manipulation needed anymore
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint, # Pass the original entrypoint
            prewarm_fnc=prewarm,
        ),
    )
