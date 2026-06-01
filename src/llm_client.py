import os
import re
import time
import logging
from dataclasses import dataclass
from typing import Optional
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

_client: Optional[OpenAI] = None


def get_client() -> OpenAI:
    global _client
    if _client is None:
        key = os.environ.get("OPENROUTER_API_KEY", "")
        if not key:
            raise EnvironmentError("OPENROUTER_API_KEY not set in environment or .env")
        _client = OpenAI(api_key=key, base_url="https://openrouter.ai/api/v1")
    return _client


@dataclass
class LLMResponse:
    content: str = ""
    reasoning: str = ""

    @property
    def full_text(self) -> str:
        if self.reasoning:
            return f"<think>\n{self.reasoning}\n</think>\n\n{self.content}"
        return self.content


def generate_full(
    messages: list,
    model: str,
    max_tokens: int = 256,
    temperature: float = 0.1,
    retries: int = 5,
    retry_delay: float = 5.0,
) -> LLMResponse:
    client = get_client()
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            msg = resp.choices[0].message
            content = msg.content or ""

            reasoning = ""
            extra = getattr(msg, "model_extra", {}) or {}
            if extra.get("reasoning"):
                reasoning = extra["reasoning"]
            elif extra.get("reasoning_content"):
                reasoning = extra["reasoning_content"]
            elif hasattr(msg, "reasoning_content") and msg.reasoning_content:
                reasoning = msg.reasoning_content

            think_match = re.search(r"<think>(.*?)</think>", content, re.DOTALL)
            if think_match and not reasoning:
                reasoning = think_match.group(1).strip()
                content = content[think_match.end():].strip()

            return LLMResponse(content=content, reasoning=reasoning)

        except Exception as exc:
            status = getattr(getattr(exc, "response", None), "status_code", None)
            is_rate_limit = status == 429 or "429" in str(exc)
            wait = (retry_delay * (2 ** attempt)) + (10 if is_rate_limit else 0)
            logger.warning("Attempt %d/%d failed for %s (wait %.1fs): %s",
                           attempt + 1, retries, model, wait, exc)
            if attempt < retries - 1:
                time.sleep(wait)
            else:
                logger.error("All retries exhausted for %s.", model)
                raise
    return LLMResponse()


def generate(messages: list, model: str, max_tokens: int = 256,
             temperature: float = 0.1) -> str:
    return generate_full(messages, model, max_tokens, temperature).content


def extract_p_yes(text: str) -> Optional[float]:
    if not text:
        return None

    m = re.search(r'"p_yes"\s*:\s*([\d]+(?:\.[\d]+)?)', text)
    if m:
        try:
            return float(max(0.0, min(1.0, float(m.group(1)))))
        except ValueError:
            pass

    m = re.search(r'p_yes\s*[=:]\s*([\d]+(?:\.[\d]+)?)', text, re.IGNORECASE)
    if m:
        try:
            return float(max(0.0, min(1.0, float(m.group(1)))))
        except ValueError:
            pass

    floats = re.findall(r'\b(0\.\d{1,4}|1\.0{1,4})\b', text)
    if floats:
        try:
            return float(floats[-1])
        except ValueError:
            pass

    logger.warning("Could not extract p_yes from: %s", text[:200])
    return None


def score_row(prompt: str, model: str, system_prompt: str = "") -> Optional[float]:
    """Run one LLM inference and return p_yes."""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    resp = generate_full(messages, model, max_tokens=128, temperature=0.1)
    return extract_p_yes(resp.full_text)
