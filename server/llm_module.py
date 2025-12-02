from typing import Iterator, Optional, List, Dict
from loguru import logger
import time

try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    logger.warning("vLLM not available")

try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    logger.warning("Ollama not available")


class LLMInference:
    """
    LLM inference module with streaming support.

    Supports:
    - Ollama (recommended for local inference)
    - vLLM (GPU-accelerated)
    - OpenAI-compatible API fallback

    Optimized for low latency with streaming token generation.
    """

    def __init__(
        self,
        model_path: str = "midm-2.0-q8_0:base",
        backend: str = "ollama",
        use_vllm: bool = False,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        max_tokens: int = 512,
        temperature: float = 0.7,
        gpu_memory_utilization: float = 0.9,
        ollama_host: Optional[str] = None,
    ):
        """
        Initialize LLM inference engine.

        Args:
            model_path: Path or name of the model (e.g., "midm-2.0-q8_0:base" for Ollama)
            backend: Backend to use ("ollama", "vllm", "openai")
            use_vllm: Whether to use vLLM (requires vLLM installed)
            api_base: API base URL for OpenAI-compatible API
            api_key: API key for OpenAI-compatible API
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            gpu_memory_utilization: GPU memory utilization for vLLM
            ollama_host: Ollama host URL (default: http://localhost:11434)
        """
        self.model_path = model_path
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.backend = backend

        # Initialize based on backend
        if backend == "ollama" and OLLAMA_AVAILABLE:
            logger.info(f"Initializing Ollama with model: {model_path}")
            self.ollama_host = ollama_host or "http://localhost:11434"

            # Create Ollama client
            self.ollama_client = ollama.Client(host=self.ollama_host)

            # Test connection and check model
            try:
                response = self.ollama_client.list()

                # Extract model names from ListResponse
                model_names = []
                if hasattr(response, 'models'):
                    # response is a ListResponse object with .models attribute
                    for model in response.models:
                        if hasattr(model, 'model'):
                            model_names.append(model.model)
                        elif hasattr(model, 'name'):
                            model_names.append(model.name)
                        elif isinstance(model, dict):
                            model_names.append(model.get('model') or model.get('name') or str(model))
                        else:
                            model_names.append(str(model))
                elif isinstance(response, dict) and 'models' in response:
                    # Fallback for dict response
                    for m in response['models']:
                        if isinstance(m, dict):
                            model_names.append(m.get('name') or m.get('model') or str(m))
                        else:
                            model_names.append(str(m))

                logger.info(f"Available Ollama models: {model_names}")

                if model_path not in model_names:
                    logger.warning(
                        f"Model '{model_path}' not found in Ollama. "
                        f"Run: ollama pull {model_path}"
                    )
            except Exception as e:
                logger.error(f"Failed to connect to Ollama: {e}")
                logger.info("Make sure Ollama is running: ollama serve")

            logger.info("Ollama backend initialized")

        elif (backend == "vllm" or use_vllm) and VLLM_AVAILABLE:
            logger.info(f"Initializing vLLM with model: {model_path}")
            start_time = time.time()

            self.llm = LLM(
                model=model_path,
                gpu_memory_utilization=gpu_memory_utilization,
                dtype="auto",
                trust_remote_code=True,
            )

            load_time = time.time() - start_time
            logger.info(f"vLLM model loaded in {load_time:.2f}s")

        elif backend == "openai":
            # OpenAI-compatible API
            try:
                from openai import OpenAI
                self.client = OpenAI(
                    base_url=api_base or "http://localhost:8000/v1",
                    api_key=api_key or "dummy-key"
                )
                logger.info(f"Using OpenAI-compatible API at {api_base}")
            except ImportError:
                raise RuntimeError("OpenAI package not available. Install with: uv pip install openai")

        else:
            raise RuntimeError(
                f"Backend '{backend}' not available or not supported. "
                f"Available: ollama={OLLAMA_AVAILABLE}, vllm={VLLM_AVAILABLE}"
            )

        # System prompt for voice assistant
        self.system_prompt = """You are a helpful voice assistant.
Provide clear, concise, and natural responses suitable for speech output.
Keep responses brief and conversational."""

        self.conversation_history: List[Dict[str, str]] = []

    def generate_streaming(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> Iterator[str]:
        """
        Generate response with streaming tokens.

        Args:
            prompt: User input prompt
            system_prompt: Override system prompt
            max_tokens: Override max tokens
            temperature: Override temperature

        Yields:
            Generated tokens as they become available
        """
        system = system_prompt or self.system_prompt
        max_tok = max_tokens or self.max_tokens
        temp = temperature or self.temperature

        # Build messages
        messages = [{"role": "system", "content": system}]

        # Add conversation history
        messages.extend(self.conversation_history)

        # Add current prompt
        messages.append({"role": "user", "content": prompt})

        if self.backend == "ollama":
            yield from self._generate_ollama_streaming(messages, max_tok, temp)
        elif self.backend == "vllm":
            yield from self._generate_vllm_streaming(messages, max_tok, temp)
        elif self.backend == "openai":
            yield from self._generate_openai_streaming(messages, max_tok, temp)

    def _generate_ollama_streaming(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int,
        temperature: float,
    ) -> Iterator[str]:
        """Generate with Ollama backend."""
        try:
            # Ollama streaming API
            response = self.ollama_client.chat(
                model=self.model_path,
                messages=messages,
                stream=True,
                options={
                    'temperature': temperature,
                    'num_predict': max_tokens,
                }
            )

            for chunk in response:
                if 'message' in chunk and 'content' in chunk['message']:
                    content = chunk['message']['content']
                    if content:
                        yield content

        except Exception as e:
            logger.error(f"Ollama API error: {e}")
            error_msg = str(e)

            if "connection" in error_msg.lower():
                yield "Error: Cannot connect to Ollama. Please make sure Ollama is running (ollama serve)."
            elif "model" in error_msg.lower():
                yield f"Error: Model '{self.model_path}' not found. Run: ollama pull {self.model_path}"
            else:
                yield f"Sorry, I encountered an error: {error_msg}"

    def _generate_vllm_streaming(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int,
        temperature: float,
    ) -> Iterator[str]:
        """Generate with vLLM backend."""
        # Format messages for the model
        # Note: Adjust formatting based on your model's chat template
        prompt_text = self._format_messages_for_model(messages)

        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=0.9,
            stop=["</s>", "<|eot_id|>"],  # Adjust for your model
        )

        # Use streaming generation
        for output in self.llm.generate(
            [prompt_text],
            sampling_params,
            use_tqdm=False
        ):
            for token_output in output.outputs:
                # Yield incremental text
                text = token_output.text
                if text:
                    yield text

    def _generate_openai_streaming(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int,
        temperature: float,
    ) -> Iterator[str]:
        """Generate with OpenAI-compatible API backend."""
        try:
            response = self.client.chat.completions.create(
                model=self.model_path,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=True,
            )

            for chunk in response:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            yield "Sorry, I encountered an error processing your request."

    def _format_messages_for_model(self, messages: List[Dict[str, str]]) -> str:
        """
        Format messages according to model's chat template.

        Adjust this based on your specific model.
        Example for Llama-3.1:
        """
        formatted = "<|begin_of_text|>"

        for msg in messages:
            role = msg["role"]
            content = msg["content"]

            if role == "system":
                formatted += f"<|start_header_id|>system<|end_header_id|>\n\n{content}<|eot_id|>"
            elif role == "user":
                formatted += f"<|start_header_id|>user<|end_header_id|>\n\n{content}<|eot_id|>"
            elif role == "assistant":
                formatted += f"<|start_header_id|>assistant<|end_header_id|>\n\n{content}<|eot_id|>"

        formatted += "<|start_header_id|>assistant<|end_header_id|>\n\n"

        return formatted

    def update_conversation(self, user_msg: str, assistant_msg: str):
        """
        Update conversation history.

        Args:
            user_msg: User message
            assistant_msg: Assistant response
        """
        self.conversation_history.append({"role": "user", "content": user_msg})
        self.conversation_history.append({"role": "assistant", "content": assistant_msg})

        # Keep only last N exchanges to prevent context overflow
        max_history = 10
        if len(self.conversation_history) > max_history * 2:
            self.conversation_history = self.conversation_history[-(max_history * 2):]

    def clear_conversation(self):
        """Clear conversation history."""
        self.conversation_history = []
