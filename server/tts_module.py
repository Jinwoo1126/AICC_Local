import numpy as np
from typing import Iterator, Optional
from loguru import logger
import re
import io
import wave


class TTSEngine:
    """
    Text-to-Speech engine with streaming support.

    Supports multiple TTS backends (Piper, StyleTTS2, etc.)
    Optimized for low-latency streaming synthesis.
    """

    def __init__(
        self,
        engine: str = "piper",
        model_path: Optional[str] = None,
        sample_rate: int = 22050,
    ):
        """
        Initialize TTS engine.

        Args:
            engine: TTS engine to use ("piper", "styletts2", "coqui")
            model_path: Path to TTS model
            sample_rate: Output sample rate
        """
        self.engine = engine
        self.sample_rate = sample_rate
        self.model_path = model_path

        if engine == "piper":
            self._init_piper()
        elif engine == "styletts2":
            self._init_styletts2()
        elif engine == "coqui":
            self._init_coqui()
        else:
            raise ValueError(f"Unsupported TTS engine: {engine}")

        logger.info(f"TTS engine initialized: {engine}")

    def _init_piper(self):
        """Initialize Piper TTS."""
        try:
            from piper import PiperVoice

            if not self.model_path:
                logger.warning(
                    "No Piper model path specified. TTS will be disabled. "
                    "Download models from: https://github.com/rhasspy/piper/releases"
                )
                self.voice = None
                return

            logger.info(f"Loading Piper model: {self.model_path}")
            try:
                self.voice = PiperVoice.load(self.model_path)
            except FileNotFoundError:
                logger.warning(
                    f"Piper model not found at: {self.model_path}. TTS will be disabled. "
                    "Download models from: https://github.com/rhasspy/piper/releases"
                )
                self.voice = None

        except ImportError:
            logger.error("Piper not installed. Install with: pip install piper-tts")
            raise

    def _init_styletts2(self):
        """Initialize StyleTTS2."""
        logger.warning("StyleTTS2 support is experimental")
        # StyleTTS2 implementation would go here
        # Requires custom installation and model setup
        raise NotImplementedError("StyleTTS2 support coming soon")

    def _init_coqui(self):
        """Initialize Coqui TTS."""
        try:
            from TTS.api import TTS

            logger.info("Loading Coqui TTS model")
            self.tts = TTS(
                model_name=self.model_path or "tts_models/en/ljspeech/tacotron2-DDC",
                gpu=False  # Set to True if GPU available
            )

        except ImportError:
            logger.error("Coqui TTS not installed. Install with: pip install TTS")
            raise

    def synthesize_streaming(
        self,
        text_stream: Iterator[str],
        chunk_size: int = 5,
    ) -> Iterator[bytes]:
        """
        Synthesize audio from streaming text.

        This is the key method for low-latency TTS.
        Instead of waiting for complete sentences, we synthesize
        as soon as we have enough tokens.

        Args:
            text_stream: Iterator yielding text chunks
            chunk_size: Number of tokens to accumulate before synthesis

        Yields:
            Audio bytes chunks
        """
        buffer = []
        word_count = 0

        for text_chunk in text_stream:
            # Add to buffer
            buffer.append(text_chunk)
            word_count += len(text_chunk.split())

            # Check if we should synthesize
            # Synthesize on:
            # 1. Sentence boundaries (. ! ?)
            # 2. Comma boundaries (,)
            # 3. Sufficient word count
            should_synthesize = (
                any(p in text_chunk for p in ['.', '!', '?', ',']) or
                word_count >= chunk_size
            )

            if should_synthesize:
                # Synthesize accumulated text
                text_to_synthesize = "".join(buffer).strip()

                if text_to_synthesize:
                    for audio_chunk in self._synthesize_chunk(text_to_synthesize):
                        yield audio_chunk

                # Clear buffer
                buffer = []
                word_count = 0

        # Synthesize any remaining text
        if buffer:
            text_to_synthesize = "".join(buffer).strip()
            if text_to_synthesize:
                for audio_chunk in self._synthesize_chunk(text_to_synthesize):
                    yield audio_chunk

    def synthesize(self, text: str) -> bytes:
        """
        Synthesize complete text to audio.

        Args:
            text: Text to synthesize

        Returns:
            Audio data as bytes (WAV format)
        """
        # Collect all chunks
        audio_chunks = list(self._synthesize_chunk(text))

        # Combine chunks
        if not audio_chunks:
            return b""

        return b"".join(audio_chunks)

    def _synthesize_chunk(self, text: str) -> Iterator[bytes]:
        """
        Synthesize a text chunk to audio.

        Args:
            text: Text to synthesize

        Yields:
            Audio bytes in WAV format
        """
        if not text.strip():
            return

        try:
            if self.engine == "piper":
                yield from self._synthesize_piper(text)
            elif self.engine == "coqui":
                yield from self._synthesize_coqui(text)
            else:
                logger.error(f"Synthesis not implemented for: {self.engine}")

        except Exception as e:
            logger.error(f"TTS synthesis error: {e}")

    def _synthesize_piper(self, text: str) -> Iterator[bytes]:
        """Synthesize with Piper."""
        if self.voice is None:
            logger.warning("Piper voice model not loaded. TTS is disabled.")
            return

        # Piper's synthesize method returns AudioChunk objects
        # Each AudioChunk has audio_int16_bytes attribute with raw audio
        audio_bytes_list = []

        for audio_chunk in self.voice.synthesize(text):
            # Get int16 audio bytes from AudioChunk
            audio_bytes_list.append(audio_chunk.audio_int16_bytes)

        # Combine all audio bytes
        audio_bytes = b"".join(audio_bytes_list)

        # Convert to WAV format
        wav_data = self._to_wav(audio_bytes, self.voice.config.sample_rate)

        yield wav_data

    def _synthesize_coqui(self, text: str) -> Iterator[bytes]:
        """Synthesize with Coqui TTS."""
        # Generate audio
        wav = self.tts.tts(text)

        # Convert numpy array to bytes
        audio_data = (np.array(wav) * 32767).astype(np.int16).tobytes()

        # Convert to WAV format
        wav_data = self._to_wav(audio_data, self.sample_rate)

        yield wav_data

    def _to_wav(self, audio_data: bytes, sample_rate: int) -> bytes:
        """
        Convert raw audio data to WAV format.

        Args:
            audio_data: Raw audio bytes
            sample_rate: Sample rate

        Returns:
            WAV formatted audio bytes
        """
        wav_buffer = io.BytesIO()

        with wave.open(wav_buffer, 'wb') as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_data)

        return wav_buffer.getvalue()

    @staticmethod
    def split_sentences(text: str) -> list[str]:
        """
        Split text into sentences for better streaming.

        Args:
            text: Input text

        Returns:
            List of sentences
        """
        # Simple sentence splitter
        sentences = re.split(r'([.!?]+\s+)', text)

        # Combine sentence with its punctuation
        result = []
        for i in range(0, len(sentences) - 1, 2):
            sentence = sentences[i] + (sentences[i + 1] if i + 1 < len(sentences) else "")
            sentence = sentence.strip()
            if sentence:
                result.append(sentence)

        # Add last part if exists
        if len(sentences) % 2 == 1 and sentences[-1].strip():
            result.append(sentences[-1].strip())

        return result
