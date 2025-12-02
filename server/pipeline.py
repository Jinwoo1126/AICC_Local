import asyncio
import numpy as np
from typing import Optional, AsyncIterator, Callable, Awaitable
from loguru import logger
from queue import Queue
import threading

from .vad_module import SileroVAD
from .stt_module_mlx import MLXWhisperSTT as FasterWhisperSTT
from .llm_module import LLMInference
from .tts_module import TTSEngine


class RealtimePipeline:
    """
    Real-time voice conversation pipeline.

    This is the core orchestrator that connects:
    Audio Input -> VAD -> STT -> LLM -> TTS -> Audio Output

    All stages are streaming for minimum latency.
    """

    def __init__(
        self,
        vad_config: dict,
        stt_config: dict,
        llm_config: dict,
        tts_config: dict,
    ):
        """
        Initialize the real-time pipeline.

        Args:
            vad_config: VAD configuration
            stt_config: STT configuration
            llm_config: LLM configuration
            tts_config: TTS configuration
        """
        logger.info("Initializing Real-time Pipeline")

        # Initialize all components
        self.vad = SileroVAD(**vad_config)
        self.stt = FasterWhisperSTT(**stt_config)
        self.llm = LLMInference(**llm_config)
        self.tts = TTSEngine(**tts_config)

        # State management
        self.is_processing = False
        self.should_interrupt = False

        # Thread-safe queue for interruption handling
        self.interrupt_event = threading.Event()

        logger.info("Pipeline initialized successfully")

    async def process_audio_stream(
        self,
        audio_iterator: AsyncIterator[bytes],
        output_callback: Callable[[bytes], Awaitable[None]],
    ):
        """
        Process streaming audio input and send audio output via callback.

        This is the main entry point for the pipeline.

        Args:
            audio_iterator: Async iterator yielding audio chunks
            output_callback: Async callback to send audio output
        """
        logger.info("Starting audio stream processing")

        try:
            async for audio_chunk in audio_iterator:
                # Check for interruption (barge-in)
                if self.should_interrupt:
                    logger.info("Barge-in detected, interrupting current processing")
                    self.should_interrupt = False
                    continue

                # Convert bytes to numpy array
                audio_data = np.frombuffer(audio_chunk, dtype=np.int16).astype(np.float32) / 32768.0

                # Process through VAD
                is_speech, speech_ended, speech_audio = self.vad.process_chunk(audio_data)

                if speech_ended and speech_audio is not None:
                    logger.info("Speech segment detected, processing...")

                    # Run the full pipeline in a separate task
                    asyncio.create_task(
                        self._process_speech_segment(speech_audio, output_callback)
                    )

        except Exception as e:
            logger.error(f"Error in audio stream processing: {e}")
            raise

    async def _process_speech_segment(
        self,
        audio: np.ndarray,
        output_callback: Callable[[bytes], Awaitable[None]],
    ):
        """
        Process a complete speech segment through STT -> LLM -> TTS.

        Args:
            audio: Speech audio segment
            output_callback: Callback to send audio output
        """
        if self.is_processing:
            logger.warning("Already processing, skipping...")
            return

        self.is_processing = True

        try:
            # Stage 1: STT (Speech-to-Text)
            logger.info("Running STT...")
            transcription = await asyncio.to_thread(
                self.stt.transcribe,
                audio
            )

            if not transcription:
                logger.warning("No transcription generated")
                self.is_processing = False
                return

            logger.info(f"Transcription: {transcription}")

            # Stage 2: LLM (Language Model)
            logger.info("Running LLM inference...")

            # Generate LLM response in thread to avoid blocking event loop
            def generate_llm_tokens():
                tokens = []
                logger.debug("Starting LLM token generation...")
                for token in self.llm.generate_streaming(transcription):
                    if self.should_interrupt:
                        logger.info("LLM generation interrupted")
                        break
                    tokens.append(token)
                    logger.debug(f"Generated token: {token}")
                return tokens

            # Run LLM generation in thread pool
            llm_tokens = await asyncio.to_thread(generate_llm_tokens)
            response_text = "".join(llm_tokens)
            logger.info(f"LLM response: {response_text}")

            if not response_text:
                logger.warning("No LLM response generated")
                self.is_processing = False
                return

            # Stage 3: TTS (Text-to-Speech)
            logger.info("Running TTS synthesis...")

            # Synthesize audio in thread to avoid blocking event loop
            def synthesize_audio():
                chunks = []
                logger.debug("Starting TTS audio synthesis...")
                for audio_chunk in self.tts.synthesize_streaming([response_text]):
                    if self.should_interrupt:
                        logger.info("TTS synthesis interrupted")
                        break
                    chunks.append(audio_chunk)
                    logger.debug(f"Generated audio chunk: {len(audio_chunk)} bytes")
                return chunks

            # Run TTS synthesis in thread pool
            audio_chunks = await asyncio.to_thread(synthesize_audio)

            # Send all audio chunks to client
            for audio_chunk in audio_chunks:
                await output_callback(audio_chunk)
            self.llm.update_conversation(transcription, response_text)

            logger.info("Speech segment processing completed")

        except Exception as e:
            logger.error(f"Error processing speech segment: {e}")

        finally:
            self.is_processing = False

    def trigger_barge_in(self):
        """
        Trigger barge-in (interrupt current processing).

        Call this when user starts speaking while assistant is talking.
        """
        logger.info("Triggering barge-in")
        self.should_interrupt = True
        self.interrupt_event.set()

    def reset(self):
        """Reset pipeline state."""
        logger.info("Resetting pipeline")
        self.vad.reset()
        self.llm.clear_conversation()
        self.is_processing = False
        self.should_interrupt = False

    async def process_text_input(
        self,
        text: str,
        output_callback: Callable[[bytes], Awaitable[None]],
    ):
        """
        Process text input directly (bypass STT).

        Useful for testing or text-based interaction.

        Args:
            text: Input text
            output_callback: Callback to send audio output
        """
        logger.info(f"Processing text input: {text}")

        # LLM generation - run in thread to avoid blocking
        def generate_llm_response():
            full_response = []
            logger.debug("Starting LLM generation...")
            for token in self.llm.generate_streaming(text):
                full_response.append(token)
                logger.debug(f"Token: {token}")
            return "".join(full_response)

        # Run LLM generation in a thread pool
        response_text = await asyncio.to_thread(generate_llm_response)
        logger.info(f"LLM response: {response_text}")

        # TTS synthesis - also run in thread to avoid blocking
        def synthesize_audio():
            chunks = []
            logger.debug("Starting TTS synthesis...")
            for audio_chunk in self.tts.synthesize_streaming([response_text]):
                chunks.append(audio_chunk)
                logger.debug(f"Generated audio chunk: {len(audio_chunk)} bytes")
            return chunks

        audio_chunks = await asyncio.to_thread(synthesize_audio)

        # Send all audio chunks to client
        for audio_chunk in audio_chunks:
            await output_callback(audio_chunk)

        # Update conversation
        self.llm.update_conversation(text, response_text)

        logger.info("Text input processing completed")
