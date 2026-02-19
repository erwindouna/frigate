"""llama.cpp Provider for Frigate AI."""

import base64
import io
import json
import logging
from typing import Any, Optional

import numpy as np
import requests
from PIL import Image

from frigate.config import GenAIProviderEnum
from frigate.genai import GenAIClient, register_genai_provider

logger = logging.getLogger(__name__)


def _to_jpeg(img_bytes: bytes) -> bytes | None:
    """Convert image bytes to JPEG. llama.cpp/STB does not support WebP."""
    try:
        img = Image.open(io.BytesIO(img_bytes))
        if img.mode != "RGB":
            img = img.convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        return buf.getvalue()
    except Exception as e:
        logger.warning("Failed to convert image to JPEG: %s", e)
        return None


@register_genai_provider(GenAIProviderEnum.llamacpp)
class LlamaCppClient(GenAIClient):
    """Generative AI client for Frigate using llama.cpp server."""

    LOCAL_OPTIMIZED_OPTIONS = {
        "temperature": 0.7,
        "repeat_penalty": 1.05,
        "top_p": 0.8,
    }

    provider: str  # base_url
    provider_options: dict[str, Any]

    def _init_provider(self):
        """Initialize the client."""
        self.provider_options = {
            **self.LOCAL_OPTIMIZED_OPTIONS,
            **self.genai_config.provider_options,
        }
        return (
            self.genai_config.base_url.rstrip("/")
            if self.genai_config.base_url
            else None
        )

    def _send(self, prompt: str, images: list[bytes]) -> Optional[str]:
        """Submit a request to llama.cpp server."""
        if self.provider is None:
            logger.warning(
                "llama.cpp provider has not been initialized, a description will not be generated. Check your llama.cpp configuration."
            )
            return None

        try:
            content = []
            for image in images:
                encoded_image = base64.b64encode(image).decode("utf-8")
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{encoded_image}",
                        },
                    }
                )
            content.append(
                {
                    "type": "text",
                    "text": prompt,
                }
            )

            # Build request payload with llama.cpp native options
            payload = {
                "messages": [
                    {
                        "role": "user",
                        "content": content,
                    },
                ],
                **self.provider_options,
            }

            response = requests.post(
                f"{self.provider}/v1/chat/completions",
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()
            result = response.json()

            if (
                result is not None
                and "choices" in result
                and len(result["choices"]) > 0
            ):
                choice = result["choices"][0]
                if "message" in choice and "content" in choice["message"]:
                    return choice["message"]["content"].strip()
            return None
        except Exception as e:
            logger.warning("llama.cpp returned an error: %s", str(e))
            return None

    def get_context_size(self) -> int:
        """Get the context window size for llama.cpp."""
        return self.genai_config.provider_options.get("context_size", 4096)

    def embed(
        self,
        texts: list[str] | None = None,
        images: list[bytes] | None = None,
    ) -> list[np.ndarray]:
        """Generate embeddings via llama.cpp /embeddings endpoint.

        Supports batch requests. Uses content format with prompt_string and
        multimodal_data for images (PR #15108). Server must be started with
        --embeddings and --mmproj for multimodal support.
        """
        if self.provider is None:
            logger.warning(
                "llama.cpp provider has not been initialized. Check your llama.cpp configuration."
            )
            return []

        texts = texts or []
        images = images or []
        if not texts and not images:
            return []

        EMBEDDING_DIM = 768

        content = []
        for text in texts:
            content.append({"prompt_string": text})
        for img in images:
            # llama.cpp uses STB which does not support WebP; convert to JPEG
            jpeg_bytes = _to_jpeg(img)
            to_encode = jpeg_bytes if jpeg_bytes is not None else img
            encoded = base64.b64encode(to_encode).decode("utf-8")
            # prompt_string must contain <__media__> placeholder for image tokenization
            content.append({
                "prompt_string": "<__media__>\n",
                "multimodal_data": [encoded],
            })

        try:
            response = requests.post(
                f"{self.provider}/embeddings",
                json={"content": content},
                timeout=self.timeout,
            )
            response.raise_for_status()
            result = response.json()

            items = result.get("data", result) if isinstance(result, dict) else result
            if not isinstance(items, list):
                logger.warning("llama.cpp embeddings returned unexpected format")
                return []

            embeddings = []
            for item in items:
                emb = item.get("embedding") if isinstance(item, dict) else None
                if emb is None:
                    logger.warning("llama.cpp embeddings item missing embedding field")
                    continue
                arr = np.array(emb, dtype=np.float32)
                orig_dim = arr.size
                if orig_dim != EMBEDDING_DIM:
                    if orig_dim > EMBEDDING_DIM:
                        arr = arr[:EMBEDDING_DIM]
                        logger.debug(
                            "Truncated llama.cpp embedding from %d to %d dimensions",
                            orig_dim,
                            EMBEDDING_DIM,
                        )
                    else:
                        arr = np.pad(
                            arr,
                            (0, EMBEDDING_DIM - orig_dim),
                            mode="constant",
                            constant_values=0,
                        )
                        logger.debug(
                            "Padded llama.cpp embedding from %d to %d dimensions",
                            orig_dim,
                            EMBEDDING_DIM,
                        )
                embeddings.append(arr)
            return embeddings
        except requests.exceptions.Timeout:
            logger.warning("llama.cpp embeddings request timed out")
            return []
        except requests.exceptions.RequestException as e:
            error_detail = str(e)
            if hasattr(e, "response") and e.response is not None:
                try:
                    error_detail = f"{str(e)} - Response: {e.response.text[:500]}"
                except Exception:
                    pass
            logger.warning("llama.cpp embeddings error: %s", error_detail)
            return []
        except Exception as e:
            logger.warning("Unexpected error in llama.cpp embeddings: %s", str(e))
            return []

    def chat_with_tools(
        self,
        messages: list[dict[str, Any]],
        tools: Optional[list[dict[str, Any]]] = None,
        tool_choice: Optional[str] = "auto",
    ) -> dict[str, Any]:
        """
        Send chat messages to llama.cpp server with optional tool definitions.

        Uses the OpenAI-compatible endpoint but passes through all native llama.cpp
        parameters (like slot_id, temperature, etc.) via provider_options.
        """
        if self.provider is None:
            logger.warning(
                "llama.cpp provider has not been initialized. Check your llama.cpp configuration."
            )
            return {
                "content": None,
                "tool_calls": None,
                "finish_reason": "error",
            }

        try:
            openai_tool_choice = None
            if tool_choice:
                if tool_choice == "none":
                    openai_tool_choice = "none"
                elif tool_choice == "auto":
                    openai_tool_choice = "auto"
                elif tool_choice == "required":
                    openai_tool_choice = "required"

            payload = {
                "messages": messages,
            }

            if tools:
                payload["tools"] = tools
                if openai_tool_choice is not None:
                    payload["tool_choice"] = openai_tool_choice

            provider_opts = {
                k: v for k, v in self.provider_options.items() if k != "context_size"
            }
            payload.update(provider_opts)

            response = requests.post(
                f"{self.provider}/v1/chat/completions",
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()
            result = response.json()

            if result is None or "choices" not in result or len(result["choices"]) == 0:
                return {
                    "content": None,
                    "tool_calls": None,
                    "finish_reason": "error",
                }

            choice = result["choices"][0]
            message = choice.get("message", {})

            content = message.get("content")
            if content:
                content = content.strip()
            else:
                content = None

            tool_calls = None
            if "tool_calls" in message and message["tool_calls"]:
                tool_calls = []
                for tool_call in message["tool_calls"]:
                    try:
                        function_data = tool_call.get("function", {})
                        arguments_str = function_data.get("arguments", "{}")
                        arguments = json.loads(arguments_str)
                    except (json.JSONDecodeError, KeyError, TypeError) as e:
                        logger.warning(
                            f"Failed to parse tool call arguments: {e}, "
                            f"tool: {function_data.get('name', 'unknown')}"
                        )
                        arguments = {}

                    tool_calls.append(
                        {
                            "id": tool_call.get("id", ""),
                            "name": function_data.get("name", ""),
                            "arguments": arguments,
                        }
                    )

            finish_reason = "error"
            if "finish_reason" in choice and choice["finish_reason"]:
                finish_reason = choice["finish_reason"]
            elif tool_calls:
                finish_reason = "tool_calls"
            elif content:
                finish_reason = "stop"

            return {
                "content": content,
                "tool_calls": tool_calls,
                "finish_reason": finish_reason,
            }

        except requests.exceptions.Timeout as e:
            logger.warning("llama.cpp request timed out: %s", str(e))
            return {
                "content": None,
                "tool_calls": None,
                "finish_reason": "error",
            }
        except requests.exceptions.RequestException as e:
            error_detail = str(e)
            if hasattr(e, "response") and e.response is not None:
                try:
                    error_body = e.response.text
                    error_detail = f"{str(e)} - Response: {error_body[:500]}"
                except Exception:
                    pass
            logger.warning("llama.cpp returned an error: %s", error_detail)
            return {
                "content": None,
                "tool_calls": None,
                "finish_reason": "error",
            }
        except Exception as e:
            logger.warning("Unexpected error in llama.cpp chat_with_tools: %s", str(e))
            return {
                "content": None,
                "tool_calls": None,
                "finish_reason": "error",
            }
