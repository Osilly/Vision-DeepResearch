"""DeepResearch reward helper using SGLang /generate text prompts."""

from __future__ import annotations

import asyncio
import os
import re
from typing import Any
from urllib.parse import urlparse

import aiohttp

ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.S | re.I)
_HTTP_SESSION: aiohttp.ClientSession | None = None


def _get_http_session(timeout: float) -> aiohttp.ClientSession:
    global _HTTP_SESSION
    if _HTTP_SESSION is None or _HTTP_SESSION.closed:
        _HTTP_SESSION = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout))
    return _HTTP_SESSION


class DeepResearchReward:
    def __init__(
        self,
        url: str | None = None,
        timeout: float | None = None,
        retry_attempts: int | None = None,
        retry_delay: float | None = None,
        temperature: float | None = None,
        max_new_tokens: int | None = None,
        backend: str = "sglang",
        openai_api_key: str | None = None,
        openai_base_url: str | None = None,
        openai_model: str | None = None,
        openai_headers: dict | None = None,
    ):
        self.judge_url = url or os.getenv("JUDGE_URL") or os.getenv("REWARD_MODEL_URL")
        self.timeout = timeout if timeout is not None else float(os.getenv("JUDGE_TIMEOUT", os.getenv("REWARD_TIMEOUT", "300")))
        self.retry_attempts = max(retry_attempts if retry_attempts is not None else int(os.getenv("JUDGE_RETRY_ATTEMPTS", os.getenv("REWARD_RETRY_ATTEMPTS", "3"))), 1)
        self.retry_delay = retry_delay if retry_delay is not None else float(os.getenv("JUDGE_RETRY_DELAY", os.getenv("REWARD_RETRY_DELAY", "1")))
        self.temperature = temperature if temperature is not None else float(os.getenv("JUDGE_TEMPERATURE", os.getenv("REWARD_TEMPERATURE", "0.1")))
        self.max_new_tokens = max_new_tokens if max_new_tokens is not None else int(os.getenv("JUDGE_MAX_NEW_TOKENS", os.getenv("REWARD_MAX_NEW_TOKENS", "300")))
        self.backend = backend
        self.openai_api_key = openai_api_key
        self.openai_base_url = openai_base_url
        self.openai_model = openai_model
        self.openai_headers = openai_headers or {}

    @staticmethod
    def normalize(text: Any) -> str:
        text = "" if text is None else str(text)
        text = text.strip().lower()
        text = re.sub(r"\s+", " ", text)
        return re.sub(r"^[^\w\-]+|[^\w\-]+$", "", text)

    @staticmethod
    def strip_think(text: str) -> str:
        text = re.sub(r"<think>[\s\S]*?</think>", " ", text or "", flags=re.I)
        return re.sub(r"\s+", " ", text).strip()

    @classmethod
    def extract_final_answer(cls, response: str) -> str:
        response = cls.strip_think(response or "")
        matches = list(ANSWER_RE.finditer(response))
        if matches:
            return matches[-1].group(1).strip()

        start = response.find("boxed{")
        if start != -1:
            i = start + 6
            depth = 1
            j = i
            while depth and j < len(response):
                depth += (response[j] == "{") - (response[j] == "}")
                j += 1
            if not depth:
                return response[i : j - 1].strip()

        return response.strip()

    @staticmethod
    def _build_prompt(question: str, answer: Any, assistant_answer: str) -> str:
        return (
            "You are an impartial judge evaluating whether a deep research report contains the correct answer.\n\n"
            f"[Question]\n{question}\n\n"
            f"[Correct Answer]\n{answer}\n\n"
            f"[Deep Research Report]\n{assistant_answer}\n\n"
            "Task: Determine if the deep research report contains the correct answer anywhere in its content.\n\n"
            "Instructions:\n"
            "1. Read through the entire research report carefully\n"
            "2. Look for the correct answer anywhere in the report (it may be embedded in paragraphs, tables, or sections)\n"
            "3. Check if the information in the report is consistent with the correct answer\n"
            '4. The answer does NOT need to be in a specific format or labeled as "final answer"\n'
            "5. Provide your reasoning\n"
            'Answer with "yes" if the report contains the correct answer, "no" if it doesn\'t or contradicts it\n\n'
            "Output format:\n"
            "correct: [yes/no]\n"
            "reasoning: [your explanation]"
        )

    @staticmethod
    def _parse_judgment(text: str) -> bool | None:
        lowered = (text or "").lower().strip()
        if "correct:" in lowered:
            for line in lowered.split("\n"):
                if "correct:" in line:
                    value = line.split("correct:", 1)[1].strip()
                    if value.startswith("yes"):
                        return True
                    if value.startswith("no"):
                        return False
                    return "yes" in value
        if lowered.startswith("correct: yes"):
            return True
        if lowered.startswith("correct: no"):
            return False
        return None

    async def _run_generate_openai(self, prompt: str) -> tuple[str | None, str | None, int]:
        try:
            import openai as _openai
        except ImportError:
            return None, "openai package not installed", 0

        client = _openai.AsyncOpenAI(
            api_key=self.openai_api_key,
            base_url=self.openai_base_url,
            default_headers=self.openai_headers,
        )
        last_error: Exception | None = None
        for attempt in range(1, self.retry_attempts + 1):
            try:
                resp = await client.chat.completions.create(
                    model=self.openai_model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    stream=False,
                )
                return resp.choices[0].message.content, None, attempt
            except Exception as exc:
                last_error = exc
                if attempt >= self.retry_attempts:
                    break
                await asyncio.sleep(self.retry_delay)
        return None, f"openai_error: {last_error}", self.retry_attempts

    def _endpoint(self) -> str | None:
        if not self.judge_url:
            return None
        url = self.judge_url.strip().rstrip("/")
        if url.endswith("/generate"):
            return url
        parsed = urlparse(url if "://" in url else f"http://{url}")
        if parsed.path and parsed.path != "/":
            return url
        return f"{url}/generate"

    async def _run_generate(self, prompt: str) -> tuple[str | None, str | None, int]:
        endpoint = self._endpoint()
        if not endpoint:
            return None, None, 0

        chat_text = (
            "<|im_start|>user\n"
            f"{prompt}\n"
            "<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        payload = {
            "text": chat_text,
            "sampling_params": {
                "temperature": self.temperature,
                "max_new_tokens": self.max_new_tokens,
                "skip_special_tokens": False,
            },
            "return_logprob": False,
        }

        last_error: Exception | None = None
        for attempt in range(1, self.retry_attempts + 1):
            try:
                session = _get_http_session(self.timeout)
                async with session.post(endpoint, json=payload) as resp:
                    resp.raise_for_status()
                    output = await resp.json()
                return str(output.get("text", "")), None, attempt
            except Exception as exc:
                last_error = exc
                if attempt >= self.retry_attempts:
                    break
                await asyncio.sleep(self.retry_delay)

        return None, f"judge_error: {last_error}", self.retry_attempts

    async def async_score(self, *, question: str, response: str, answer: Any) -> dict[str, Any]:
        extracted = self.extract_final_answer(response)
        refs = [self.normalize(item) for item in answer] if isinstance(answer, (list, tuple)) else [self.normalize(answer)]
        pred = self.normalize(extracted)
        exact_match = pred in refs

        metadata: dict[str, Any] = {
            "extracted_answer": extracted,
            "normalized_prediction": pred,
            "normalized_refs": refs,
            "exact_match": exact_match,
            "judge_used": False,
            "judge_decided": False,
            "judge_attempts": 0,
            "judgment": None,
            "judge_endpoint_style": "sglang_generate_text",
        }

        if answer is None:
            metadata["error"] = "No answer provided"
            return {"reward": 0.0, "is_correct": False, "metadata": metadata}

        if not extracted or extracted.strip().lower() == "null":
            metadata["error"] = "Extracted answer is null or empty"
            return {"reward": 0.0, "is_correct": False, "metadata": metadata}

        prompt = self._build_prompt(question, answer, extracted)
        if self.backend == "openai":
            judgment, error, attempts = await self._run_generate_openai(prompt)
        else:
            judgment, error, attempts = await self._run_generate(prompt)
        parsed = self._parse_judgment(judgment or "") if judgment is not None else None
        metadata.update(
            {
                "judge_used": judgment is not None or error is not None,
                "judge_decided": parsed is not None,
                "judge_attempts": attempts,
                "judgment": judgment if judgment is not None else error,
            }
        )

        if parsed is None:
            metadata["fallback_reason"] = "judge_unavailable_or_undecided_use_exact_match"
            is_correct = exact_match
        else:
            is_correct = bool(parsed)

        return {"reward": 1.0 if is_correct else 0.0, "is_correct": is_correct, "metadata": metadata}
