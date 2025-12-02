import torch
import numpy as np
from typing import Optional, Tuple
from loguru import logger


class SileroVAD:
    """
    Silero VAD wrapper for real-time voice activity detection.

    This module detects speech segments in audio streams with very low latency.
    It uses Silero VAD model which is optimized for speed and accuracy.
    """

    def __init__(
        self,
        threshold: float = 0.5,
        sample_rate: int = 16000,
        frame_ms: int = 30,
        min_silence_duration_ms: int = 500
    ):
        """
        Initialize Silero VAD.

        Args:
            threshold: Voice probability threshold (0.0 - 1.0)
            sample_rate: Audio sample rate in Hz
            frame_ms: Frame duration in milliseconds (10, 20, or 30)
            min_silence_duration_ms: Minimum silence duration to consider speech end
        """
        self.threshold = threshold
        self.sample_rate = sample_rate
        self.frame_ms = frame_ms
        self.min_silence_duration_ms = min_silence_duration_ms

        # Calculate frame size in samples
        self.frame_size = int(sample_rate * frame_ms / 1000)
        self.min_silence_frames = int(min_silence_duration_ms / frame_ms)

        # Load Silero VAD model
        logger.info("Loading Silero VAD model...")
        self.model, _ = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False,
            onnx=False
        )
        self.model.eval()

        # State tracking
        self.reset()

        logger.info(f"VAD initialized (threshold={threshold}, frame_ms={frame_ms})")

    def reset(self):
        """Reset VAD state."""
        self._h = None
        self._c = None
        self.silence_frames = 0
        self.is_speaking = False
        self.speech_buffer = []

    def process_chunk(self, audio_chunk: np.ndarray) -> Tuple[bool, bool, Optional[np.ndarray]]:
        """
        Process audio chunk and detect voice activity.

        Args:
            audio_chunk: Audio data as numpy array (float32, [-1, 1])

        Returns:
            Tuple of (is_speech, speech_ended, speech_audio)
            - is_speech: True if current chunk contains speech
            - speech_ended: True if a complete speech segment has ended
            - speech_audio: Complete speech segment if speech_ended is True
        """
        # Ensure correct shape and type
        if audio_chunk.dtype != np.float32:
            audio_chunk = audio_chunk.astype(np.float32)

        # Silero VAD requires exactly 512 samples for 16kHz (or 256 for 8kHz)
        required_samples = 512 if self.sample_rate == 16000 else 256

        # Pad or trim to required size
        if len(audio_chunk) < required_samples:
            audio_chunk = np.pad(audio_chunk, (0, required_samples - len(audio_chunk)))
        elif len(audio_chunk) > required_samples:
            audio_chunk = audio_chunk[:required_samples]

        # Convert to tensor
        audio_tensor = torch.from_numpy(audio_chunk).unsqueeze(0)

        # Get speech probability
        with torch.no_grad():
            speech_prob = self.model(audio_tensor, self.sample_rate).item()

        # Determine if current chunk is speech
        is_speech = speech_prob >= self.threshold
        speech_ended = False
        speech_audio = None

        if is_speech:
            # Speech detected
            self.silence_frames = 0
            if not self.is_speaking:
                logger.debug("Speech started")
                self.is_speaking = True
            self.speech_buffer.append(audio_chunk)
        else:
            # No speech detected
            if self.is_speaking:
                self.silence_frames += 1
                self.speech_buffer.append(audio_chunk)

                # Check if silence duration exceeds threshold
                if self.silence_frames >= self.min_silence_frames:
                    logger.debug("Speech ended")
                    speech_ended = True
                    speech_audio = np.concatenate(self.speech_buffer)

                    # Reset state
                    self.is_speaking = False
                    self.silence_frames = 0
                    self.speech_buffer = []

        return is_speech, speech_ended, speech_audio

    def finalize_speech(self) -> Optional[np.ndarray]:
        """
        Finalize any remaining speech in buffer.
        Useful when stream ends.

        Returns:
            Remaining speech audio or None
        """
        if len(self.speech_buffer) > 0:
            speech_audio = np.concatenate(self.speech_buffer)
            self.speech_buffer = []
            self.is_speaking = False
            return speech_audio
        return None
