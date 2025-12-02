import numpy as np
from faster_whisper import WhisperModel
from typing import Optional, Iterator
from loguru import logger
import time


class FasterWhisperSTT:
    """
    Faster-Whisper based Speech-to-Text module.

    This module uses CTranslate2 backend for fast inference.
    Supports streaming transcription for real-time applications.
    """

    def __init__(
        self,
        model_size: str = "base",
        device: str = "cuda",
        compute_type: str = "int8",
        language: str = "ko",  # Korean by default
        beam_size: int = 1,  # beam_size=1 for faster inference
    ):
        """
        Initialize Faster-Whisper model.

        Args:
            model_size: Model size (tiny, base, small, medium, large-v3)
            device: Device to use (cuda, cpu)
            compute_type: Computation type (int8, float16, float32)
            language: Language code (ko, en, etc.)
            beam_size: Beam size for decoding (1 for greedy, faster)
        """
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.language = language
        self.beam_size = beam_size

        logger.info(f"Loading Faster-Whisper model: {model_size}")
        start_time = time.time()

        self.model = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type
        )

        load_time = time.time() - start_time
        logger.info(f"Whisper model loaded in {load_time:.2f}s")

    def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
        language: Optional[str] = None,
    ) -> str:
        """
        Transcribe audio to text.

        Args:
            audio: Audio data as numpy array (float32)
            sample_rate: Sample rate of audio
            language: Language override (uses default if None)

        Returns:
            Transcribed text
        """
        if language is None:
            language = self.language

        # Ensure audio is float32
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        # Transcribe
        start_time = time.time()

        segments, info = self.model.transcribe(
            audio,
            language=language,
            beam_size=self.beam_size,
            vad_filter=False,  # VAD already handled externally
            word_timestamps=False,  # Faster without word timestamps
        )

        # Collect all segments
        text_parts = []
        for segment in segments:
            text_parts.append(segment.text.strip())

        transcription = " ".join(text_parts).strip()

        inference_time = time.time() - start_time
        logger.debug(
            f"Transcribed in {inference_time:.2f}s "
            f"(detected language: {info.language}, "
            f"probability: {info.language_probability:.2f})"
        )

        return transcription

    def transcribe_streaming(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
        language: Optional[str] = None,
    ) -> Iterator[str]:
        """
        Streaming transcription that yields text as it becomes available.

        Args:
            audio: Audio data as numpy array (float32)
            sample_rate: Sample rate of audio
            language: Language override (uses default if None)

        Yields:
            Text segments as they are transcribed
        """
        if language is None:
            language = self.language

        # Ensure audio is float32
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        # Transcribe with streaming
        segments, _ = self.model.transcribe(
            audio,
            language=language,
            beam_size=self.beam_size,
            vad_filter=False,
            word_timestamps=False,
        )

        # Yield segments as they arrive
        for segment in segments:
            text = segment.text.strip()
            if text:
                yield text

    def detect_language(self, audio: np.ndarray) -> tuple[str, float]:
        """
        Detect language from audio.

        Args:
            audio: Audio data as numpy array (float32)

        Returns:
            Tuple of (language_code, probability)
        """
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        segments, info = self.model.transcribe(
            audio,
            beam_size=1,
            vad_filter=False,
        )

        # Consume one segment to get language info
        _ = next(segments, None)

        return info.language, info.language_probability
