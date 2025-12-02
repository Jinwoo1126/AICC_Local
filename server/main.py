import os
import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger
from dotenv import load_dotenv
import sys
from pathlib import Path

from .pipeline import RealtimePipeline

# Load environment variables
load_dotenv()

# Configure logging
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
    level="DEBUG"  # Changed to DEBUG for detailed logging
)

# Initialize FastAPI app
app = FastAPI(
    title="Real-time Voice Assistant",
    description="Low-latency voice conversation system",
    version="1.0.0"
)

# Mount static files for client
client_dir = Path(__file__).parent.parent / "client"
if client_dir.exists():
    app.mount("/static", StaticFiles(directory=str(client_dir)), name="static")


# Global pipeline instance (initialized on startup)
pipeline: RealtimePipeline = None


@app.on_event("startup")
async def startup_event():
    """Initialize pipeline on server startup."""
    global pipeline

    logger.info("Starting Real-time Voice Assistant Server")

    # Load configuration from environment
    vad_config = {
        "threshold": float(os.getenv("VAD_THRESHOLD", "0.5")),
        "sample_rate": int(os.getenv("SAMPLE_RATE", "16000")),
        "frame_ms": int(os.getenv("VAD_FRAME_MS", "30")),
        "min_silence_duration_ms": int(os.getenv("SILENCE_DURATION_MS", "500")),
    }

    stt_config = {
        "model_size": os.getenv("WHISPER_MODEL", "base"),
        "device": os.getenv("WHISPER_DEVICE", "cpu"),
        "compute_type": os.getenv("WHISPER_COMPUTE_TYPE", "int8"),
        "language": "en",  # Using English mode as workaround for MLX Whisper Korean issue
        "beam_size": 1,
    }

    llm_config = {
        "model_path": os.getenv("LLM_MODEL_PATH", "midm-2.0-q8_0:base"),
        "backend": os.getenv("LLM_BACKEND", "ollama"),
        "ollama_host": os.getenv("OLLAMA_HOST", "http://localhost:11434"),
        "use_vllm": os.getenv("LLM_BACKEND", "ollama") == "vllm",
        "api_base": os.getenv("OPENAI_API_BASE"),
        "api_key": os.getenv("OPENAI_API_KEY"),
        "max_tokens": int(os.getenv("LLM_MAX_TOKENS", "512")),
        "temperature": float(os.getenv("LLM_TEMPERATURE", "0.7")),
        "gpu_memory_utilization": float(os.getenv("LLM_GPU_MEMORY_UTILIZATION", "0.9")),
    }

    tts_config = {
        "engine": os.getenv("TTS_ENGINE", "piper"),
        "model_path": os.getenv("PIPER_MODEL_PATH"),
        "sample_rate": int(os.getenv("TTS_SAMPLE_RATE", "22050")),
    }

    try:
        pipeline = RealtimePipeline(
            vad_config=vad_config,
            stt_config=stt_config,
            llm_config=llm_config,
            tts_config=tts_config,
        )
        logger.info("Pipeline initialized successfully")

    except Exception as e:
        logger.error(f"Failed to initialize pipeline: {e}")
        raise


@app.get("/")
async def root():
    """Root endpoint with basic info."""
    return {
        "name": "Real-time Voice Assistant",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "websocket": "/ws",
            "health": "/health",
            "client": "/client"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "pipeline_ready": pipeline is not None,
    }


@app.get("/client")
async def get_client():
    """Serve the web client."""
    client_file = Path(__file__).parent.parent / "client" / "index.html"

    if not client_file.exists():
        return HTMLResponse(
            content="<h1>Client not found</h1><p>Please create client/index.html</p>",
            status_code=404
        )

    with open(client_file, "r") as f:
        return HTMLResponse(content=f.read())


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time voice conversation.

    Protocol:
    - Client sends: Audio chunks (binary) or control messages (JSON)
    - Server sends: Audio chunks (binary) or status messages (JSON)

    Control messages:
    - {"type": "barge_in"}: Interrupt current processing
    - {"type": "reset"}: Reset conversation
    - {"type": "text", "content": "..."}: Text input (bypass STT)
    """
    await websocket.accept()
    logger.info(f"WebSocket connection established: {websocket.client}")

    # Create audio output callback
    async def send_audio(audio_chunk: bytes):
        """Send audio chunk to client."""
        try:
            await websocket.send_bytes(audio_chunk)
        except Exception as e:
            logger.error(f"Error sending audio: {e}")

    # Audio stream async generator
    async def audio_stream():
        """Async generator for incoming audio chunks."""
        try:
            while True:
                data = await websocket.receive()

                # Handle binary audio data
                if "bytes" in data:
                    yield data["bytes"]

                # Handle text control messages
                elif "text" in data:
                    import json
                    try:
                        message = json.loads(data["text"])
                        message_type = message.get("type")

                        if message_type == "barge_in":
                            logger.info("Barge-in requested")
                            pipeline.trigger_barge_in()

                        elif message_type == "reset":
                            logger.info("Reset requested")
                            pipeline.reset()
                            await websocket.send_json({"type": "reset_complete"})

                        elif message_type == "text":
                            content = message.get("content", "")
                            logger.info(f"Text input: {content}")
                            # Process text input
                            await pipeline.process_text_input(content, send_audio)

                        elif message_type == "ping":
                            await websocket.send_json({"type": "pong"})

                    except json.JSONDecodeError:
                        logger.warning("Invalid JSON message received")

        except WebSocketDisconnect:
            logger.info("WebSocket disconnected")
        except Exception as e:
            logger.error(f"Error in audio stream: {e}")

    try:
        # Start processing audio stream
        await pipeline.process_audio_stream(audio_stream(), send_audio)

    except WebSocketDisconnect:
        logger.info(f"Client disconnected: {websocket.client}")

    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        try:
            await websocket.close(code=1011, reason="Internal server error")
        except:
            pass


if __name__ == "__main__":
    import uvicorn

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))

    logger.info(f"Starting server on {host}:{port}")

    uvicorn.run(
        "server.main:app",
        host=host,
        port=port,
        reload=False,
        log_level="info"
    )
