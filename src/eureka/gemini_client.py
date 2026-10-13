"""Gemini client with Eureka-compatible prompting and retry/backoff."""

from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.exceptions import GeminiAPIError, GeminiTimeoutError

_EUREKA_LLM = Path(__file__).resolve().parents[2] / "third_party" / "IsaacLabEureka" / "source" / "isaaclab_eureka"
if str(_EUREKA_LLM) not in sys.path:
    sys.path.insert(0, str(_EUREKA_LLM))

from isaaclab_eureka.managers.llm_manager import LLMManager  # noqa: E402
from isaaclab_eureka.config.prompt_templates import DIRECT_WORKFLOW_INITIAL_PROMPT  # noqa: E402


CODE_BLOCK_RE = re.compile(r"```python(.*?)```", re.DOTALL)


class GeminiClient:
    """Thin wrapper around Isaac Lab Eureka's LLMManager, forced onto Gemini."""

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-2.0-flash",
        timeout: int = 60,
        temperature: float = 1.0,
        num_suggestions: int = 1,
        system_prompt: Optional[str] = None,
    ):
        if not api_key:
            raise GeminiAPIError("GEMINI_API_KEY is missing or empty.")
        os.environ["GEMINI_API_KEY"] = api_key
        os.environ["LLM_MODEL"] = model
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.temperature = temperature
        self.num_suggestions = num_suggestions
        self.system_prompt = system_prompt or DIRECT_WORKFLOW_INITIAL_PROMPT
        self._manager = LLMManager(
            gpt_model=model,
            num_suggestions=num_suggestions,
            temperature=temperature,
            system_prompt=self.system_prompt,
        )
        self.last_raw_outputs: List[str] = []
        self.last_reward_strings: List[str] = []
        self.last_latency_s: float = 0.0
        self.request_log: List[Dict[str, Any]] = []

    def generate_reward_function(self, user_prompt: str, assistant_prompt: Optional[str] = None) -> Dict[str, Any]:
        started = time.perf_counter()
        try:
            result = self._retry_with_backoff(lambda: self._manager.prompt(user_prompt, assistant_prompt))
        except TimeoutError as exc:
            raise GeminiTimeoutError(str(exc)) from exc
        except Exception as exc:
            raise GeminiAPIError(str(exc)) from exc
        self.last_latency_s = time.perf_counter() - started
        self.last_raw_outputs = result.get("raw_outputs") or []
        self.last_reward_strings = result.get("reward_strings") or []
        record = {
            "latency_s": self.last_latency_s,
            "model": self.model,
            "user_prompt": user_prompt,
            "assistant_prompt": assistant_prompt,
            "raw_outputs": self.last_raw_outputs,
            "reward_strings": self.last_reward_strings,
        }
        self.request_log.append(record)
        return result

    def extract_code(self, response: str) -> str:
        matches = CODE_BLOCK_RE.findall(response or "")
        return matches[-1].lstrip("\n") if matches else ""

    def _retry_with_backoff(self, func, max_retries: int = 3):
        delay = 1.0
        last_exc: Optional[Exception] = None
        for attempt in range(max_retries + 1):
            try:
                return func()
            except Exception as exc:
                last_exc = exc
                if attempt >= max_retries:
                    break
                time.sleep(min(delay, 8.0))
                delay *= 2
        raise GeminiAPIError(f"Gemini call failed after retries: {last_exc}") from last_exc
