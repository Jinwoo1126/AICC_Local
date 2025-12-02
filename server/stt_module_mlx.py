import numpy as np
from typing import Optional, Iterator
from loguru import logger
import time
import mlx_whisper
import tempfile
import soundfile as sf
import os


class MLXWhisperSTT:
    """
    MLX-Whisper based Speech-to-Text module for Apple Silicon.

    This module uses Apple's MLX framework for fast inference on Mac.
    Optimized for M1/M2/M3 chips.
    """

    def __init__(
        self,
        model_size: str = "base",
        language: str = "ko",  # Korean by default
        device: str = "cpu",  # Ignored for MLX (always uses Apple Silicon GPU)
        compute_type: str = "int8",  # Ignored for MLX
        beam_size: int = 1,  # Ignored for MLX
    ):
        """
        Initialize MLX-Whisper model.

        Args:
            model_size: Model size (tiny, base, small, medium, large-v3)
            language: Language code (ko, en, etc.)
            device: Ignored (MLX uses Apple Silicon GPU automatically)
            compute_type: Ignored (MLX handles precision automatically)
            beam_size: Ignored (MLX uses default beam search)
        """
        self.model_size = model_size
        self.language = language

        # Log that we're using MLX
        if device == "cuda":
            logger.info("Note: CUDA device requested but using MLX (Apple Silicon GPU) instead")

        # Map model sizes to HuggingFace MLX model paths
        model_map = {
            "tiny": "mlx-community/whisper-tiny-mlx",
            "base": "mlx-community/whisper-base-mlx",
            "small": "mlx-community/whisper-small-mlx",
            "medium": "mlx-community/whisper-medium-mlx",
            "large": "mlx-community/whisper-large-v3-mlx",
            "large-v3": "mlx-community/whisper-large-v3-mlx",
        }

        self.model_path = model_map.get(model_size, "mlx-community/whisper-base-mlx")

        logger.info(f"Loading MLX-Whisper model: {self.model_path}")
        start_time = time.time()

        # Pre-load the model to avoid loading it on every transcription
        try:
            # Download/cache the model by doing a dummy transcription
            logger.info("Pre-loading model (this may take a moment on first run)...")
            # Create a dummy audio file to trigger model download
            dummy_audio = np.zeros(16000, dtype=np.float32)  # 1 second of silence
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                tmp_path = tmp_file.name
                sf.write(tmp_path, dummy_audio, 16000)

            try:
                _ = mlx_whisper.transcribe(tmp_path, path_or_hf_repo=self.model_path, verbose=False)
            finally:
                try:
                    os.unlink(tmp_path)
                except:
                    pass

            logger.info("Model pre-loaded successfully")
        except Exception as e:
            logger.warning(f"Could not pre-load model (will load on first use): {e}")

        load_time = time.time() - start_time
        logger.info(f"MLX-Whisper initialized in {load_time:.2f}s")

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

        # Normalize audio to [-1, 1] range if needed
        max_val = np.abs(audio).max()
        if max_val > 1.0:
            logger.warning(f"Audio values exceed [-1, 1] range (max: {max_val}), normalizing...")
            audio = audio / max_val

        # Check audio stats
        audio_duration = len(audio) / sample_rate
        audio_rms = np.sqrt(np.mean(audio ** 2))
        logger.info(
            f"Transcribing audio: {len(audio)} samples, {audio_duration:.2f}s, "
            f"sample_rate={sample_rate}, RMS={audio_rms:.4f}, max={np.abs(audio).max():.4f}"
        )

        if len(audio) < 1600:  # Less than 0.1s at 16kHz
            logger.warning("Audio too short for transcription")
            return ""

        # Transcribe
        start_time = time.time()

        try:
            # MLX Whisper expects a file path, so save audio to temp file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                tmp_path = tmp_file.name
                logger.debug(f"Saving audio to temp file: {tmp_path}")
                sf.write(tmp_path, audio, sample_rate)

                # Verify file was written
                file_size = os.path.getsize(tmp_path)
                logger.debug(f"Temp file size: {file_size} bytes")

            try:
                # Force Korean language for better transcription results
                logger.debug(f"Starting MLX Whisper transcription with language={language}...")
                result = mlx_whisper.transcribe(
                    tmp_path,
                    path_or_hf_repo=self.model_path,
                    language=language,  # Use specified language (Korean by default)
                    word_timestamps=False,  # Faster without word timestamps
                    verbose=False,  # Suppress MLX Whisper logs
                )

                transcription = result["text"].strip()
                inference_time = time.time() - start_time

                logger.info(
                    f"Transcribed in {inference_time:.2f}s: '{transcription}' "
                    f"(language: {result.get('language', 'unknown')})"
                )

                return transcription

            finally:
                # Clean up temp file
                try:
                    os.unlink(tmp_path)
                    logger.debug(f"Cleaned up temp file: {tmp_path}")
                except Exception as cleanup_err:
                    logger.warning(f"Could not delete temp file {tmp_path}: {cleanup_err}")

        except Exception as e:
            logger.error(f"Transcription error: {e}", exc_info=True)
            return ""

    def transcribe_streaming(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
        language: Optional[str] = None,
    ) -> Iterator[str]:
        """
        Streaming transcription that yields text as it becomes available.

        Note: MLX Whisper doesn't support true streaming, so we return
        the full transcription as a single chunk.

        Args:
            audio: Audio data as numpy array (float32)
            sample_rate: Sample rate of audio
            language: Language override (uses default if None)

        Yields:
            Text segments as they are transcribed
        """
        text = self.transcribe(audio, sample_rate, language)
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

        try:
            result = mlx_whisper.transcribe(
                audio,
                path_or_hf_repo=self.model_path,
            )

            language = result.get("language", self.language)
            # MLX Whisper doesn't provide probability, so we return 1.0
            return language, 1.0

        except Exception as e:
            logger.error(f"Language detection error: {e}")
            return self.language, 1.0


# Alias for backward compatibility
FasterWhisperSTT = MLXWhisperSTT
