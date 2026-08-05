"""
Visit tool for Vision-DeepResearch multi-turn tool-calling rollout.

Based on vision_deepresearch_async_workflow/tools/visit_tool.py
"""

import asyncio
import json
import os
import random
from typing import Any

from examples.vision_deepresearch.tools.shared import (
    DeepResearchTool,
    call_extract_model_async,
    log_tool_event,
    parse_json_from_model_output,
    run_with_retries_async,
)


class VisitTool(DeepResearchTool):
    """Web page visiting with content extraction."""

    MAX_URLS = 5
    MAX_CONTENT_CHARS = 120000

    EXTRACTOR_PROMPT = """Please process the following webpage content and user goal to extract relevant information:

## **Webpage Content** 
{webpage_content}

## **User Goal**
{goal}

## **Task Guidelines**
1. **Content Scanning for Rational**: Locate the **specific sections/data** directly related to the user's goal within the webpage content
2. **Key Extraction for Evidence**: Identify and extract the **most relevant information** from the content, you never miss any important information, output the **full original context** of the content as far as possible, it can be more than three paragraphs.
3. **Summary Output for Summary**: Organize into a concise paragraph with logical flow, prioritizing clarity and judge the contribution of the information to the goal.

**Final Output Requirements**
- Return a valid JSON object only (no code fences, Markdown, comments, or additional text).
- The JSON must contain exactly the keys "rational", "evidence", and "summary".
- Each key must map to a string value. Use an empty string if no content is available.
- Do not include any extra keys or explanatory sentences outside the JSON object.

Example:
{{"rational": "Explain why the information is relevant to the goal.", "evidence": "Quote or paraphrase the key supporting content from the webpage.", "summary": "Provide a concise summary that connects the evidence back to the goal."}}
"""

    def __init__(self):
        super().__init__(
            name="visit",
            description="Visit webpage(s) and return the summary of the content.",
            parameters={
                "type": "object",
                "properties": {
                    "url": {
                        "type": ["string", "array"],
                        "items": {"type": "string"},
                        "minItems": 1,
                        "description": "The URL(s) of the webpage(s) to visit. Can be a single URL or an array of URLs.",
                    },
                    "goal": {
                        "type": "string",
                        "description": "The goal of the visit for webpage(s).",
                    },
                },
                "required": ["url", "goal"],
            },
        )
        self.zhipu_api_key = os.getenv("ZHIPU_API_KEY")
        self.jina_api_key = os.getenv("JINA_API_KEY")
        self.zhipu_reader_url = os.getenv(
            "READER_URL", "https://open.bigmodel.cn/api/paas/v4/reader"
        )
        self.jina_reader_url = os.getenv("READER_URL", "https://r.jina.ai")
        self.extract_model = os.getenv("EXTRACT_MODEL", "Qwen3-VL-30B-A3B-Instruct")
        self.extract_max_tokens = 16384
        raw_extract_urls = os.getenv("EXTRACT_URL", "")
        self.extract_urls = [
            item.strip() for item in raw_extract_urls.split(",") if item.strip()
        ]

    async def call(self, url: str | list, goal: str = "", **kwargs) -> str:
        """Visit webpages via Reader API and optionally summarize with a local model."""
        urls = [url] if isinstance(url, str) else url
        if not urls:
            return "[Visit] No valid URL provided"

        tasks = [
            self._handle_single_url(target_url, goal)
            for target_url in urls[: self.MAX_URLS]
        ]
        results = await asyncio.gather(*tasks) if tasks else []

        return "\n\n=======\n\n".join(results)

    async def _handle_single_url(self, url: str, goal: str) -> str:
        normalized_url = self._normalize_url(url)

        try:
            reader_payload = await self._fetch_reader_content(normalized_url)
        except Exception as exc:
            log_tool_event(
                source="Visit/Reader",
                status="Exception",
                message=f"url={normalized_url}",
                error=str(exc),
                level="ERROR",
            )
            return self._build_failure_message(
                normalized_url, goal, f"Unable to fetch webpage content: {exc}"
            )

        if reader_payload is None:
            return self._build_failure_message(
                normalized_url, goal, "Reader API returned empty payload"
            )

        content = reader_payload.get("content") or ""
        description = reader_payload.get("description") or ""

        if not content:
            fallback = description or "Webpage content is empty"
            return self._build_failure_message(normalized_url, goal, fallback)

        content = self._truncate_content(content)

        summary_result = await self._summarize_with_extract(content, goal, reader_payload)

        if summary_result is None:
            log_tool_event(
                "Visit", "ExtractSummaryFailed", f"url={normalized_url}", level="ERROR"
            )
            return self._build_summary_failure_message(
                normalized_url,
                goal,
                "Summary service is unavailable or failed to produce a valid summary.",
            )
        else:
            evidence_text = summary_result.get("evidence") or "Summary service returned no evidence."
            summary_text = summary_result.get("summary") or "Summary service returned no summary."

        return self._format_success(normalized_url, goal, evidence_text, summary_text)

    def _normalize_url(self, url: str) -> str:
        from urllib.parse import urlparse
        parsed = urlparse(url)
        if not parsed.scheme:
            return f"https://{url}"
        return url

    def _select_extract_url(self) -> str | None:
        if not self.extract_urls:
            return None
        return random.choice(self.extract_urls)

    def _build_zhipu_reader_headers(self) -> dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "Authorization": self.zhipu_api_key or "",
            "X-Return-Format": "markdown",
            "X-No-Cache": "false",
            "X-Timeout": "60",
            "X-Retain-Images": "false",
            "X-With-Images-Summary": "false",
            "X-With-Links-Summary": "false",
        }
        reader_source = os.getenv("READER_SOURCE")
        if reader_source:
            headers["X-Source"] = reader_source
        return headers

    def _parse_zhipu_reader_payload(self, payload: dict[str, Any], url: str) -> dict[str, Any]:
        if not isinstance(payload, dict):
            raise RuntimeError("Reader API payload structure is invalid")

        payload_format = ""
        data = None
        reader_result = payload.get("reader_result")
        if isinstance(reader_result, dict):
            data = reader_result
            payload_format = "reader_result"
        elif payload.get("code") == 200 and isinstance(payload.get("data"), dict):
            data = payload["data"]
            payload_format = "code_data"
        elif isinstance(payload.get("data"), dict) and (
            "content" in payload["data"] or "description" in payload["data"]
        ):
            data = payload["data"]
            payload_format = "data"
        else:
            raise RuntimeError(
                "Reader API returned unsupported payload: "
                f"keys={list(payload.keys())}, code={payload.get('code')}, "
                f"message={payload.get('msg') or payload.get('message') or payload.get('error')}"
            )

        meta = dict(data)
        meta.setdefault("provider", "zhipu")
        meta.setdefault("payload_format", payload_format)
        meta.setdefault("reader_url", self.zhipu_reader_url)
        for key in ("id", "request_id", "model"):
            if payload.get(key) is not None:
                meta.setdefault(key, payload[key])

        return {
            "content": data.get("content") or "",
            "description": data.get("description") or "",
            "meta": meta,
        }

    async def _fetch_reader_content(self, url: str) -> dict[str, Any] | None:
        try:
            import requests
        except ImportError as exc:
            raise RuntimeError("Visit tool requires 'requests' package") from exc

        proxies = self._get_requests_proxies()

        if self.zhipu_api_key:
            headers = self._build_zhipu_reader_headers()
            body = {"url": url}

            def send_request():
                return requests.post(
                    self.zhipu_reader_url,
                    headers=headers,
                    json=body,
                    timeout=60,
                    proxies=proxies,
                )

            response = await run_with_retries_async(send_request, executor=self.executor)

            if response.status_code != 200:
                raise RuntimeError(
                    f"Reader API returned HTTP {response.status_code}: {response.text[:1000]}"
                )

            try:
                payload = response.json()
            except json.JSONDecodeError as exc:
                raise RuntimeError("Reader API returned non-JSON payload") from exc

            result = self._parse_zhipu_reader_payload(payload, url)
        else:
            headers = {"Authorization": self.jina_api_key}
            body = {"url": url}

            def send_request():
                return requests.post(
                    self.jina_reader_url,
                    headers=headers,
                    data=body,
                    timeout=60,
                    proxies=proxies,
                )

            response = await run_with_retries_async(send_request, executor=self.executor)

            if response.status_code != 200:
                raise RuntimeError(f"Reader API returned HTTP {response.status_code}")

            result = {
                "content": response.text or "",
                "description": "",
                "meta": {"provider": "jina", "url": url, "reader_url": self.jina_reader_url},
            }

        return result

    def _truncate_content(self, content: str) -> str:
        if len(content) <= self.MAX_CONTENT_CHARS:
            return content
        return content[: self.MAX_CONTENT_CHARS] + "\n[Content truncated...]"

    async def _summarize_with_extract(
        self, content: str, goal: str, reader_payload: dict[str, Any]
    ) -> dict[str, Any] | None:
        extract_url = self._select_extract_url()
        if not extract_url:
            log_tool_event(
                source="Visit/Extract",
                status="Config",
                message="EXTRACT_URL is not set, skip extract service",
            )
            return None

        try:
            import requests
        except ImportError:
            log_tool_event(
                source="Visit/Extract",
                status="DependencyMissing",
                message="'requests' package not installed",
                level="WARNING",
            )
            return None

        prompt = self.EXTRACTOR_PROMPT.format(
            webpage_content=content, goal=goal or "N/A"
        )

        extract_messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]

        raw_payload = await call_extract_model_async(
            url=extract_url,
            model=self.extract_model,
            messages=extract_messages,
            max_tokens=self.extract_max_tokens,
            proxies=self._get_requests_proxies(),
            executor=self.executor,
            source="Visit/Extract",
        )
        if raw_payload is None:
            return None

        content_dict = parse_json_from_model_output(raw_payload or "")
        if content_dict is None:
            log_tool_event(
                source="Visit/Extract",
                status="ParseFailed",
                message=raw_payload,
                level="WARNING",
            )
            return None

        return content_dict

    def _build_failure_message(self, url: str, goal: str, reason: str) -> str:
        useful_information = f"The useful information in {url} for user goal {goal or 'N/A'} as follows: \n\n"
        useful_information += "Evidence in page: \n" + reason + "\n\n"
        useful_information += "Summary: \nUnable to retrieve webpage content. Please check the link or try again later.\n\n"
        return useful_information

    def _build_summary_failure_message(self, url: str, goal: str, reason: str) -> str:
        useful_information = f"The useful information in {url} for user goal {goal or 'N/A'} as follows: \n\n"
        useful_information += "Evidence in page: \nUnable to summarize webpage content.\n\n"
        useful_information += "Summary: \n" + reason + " Please try another source or visit the page again later.\n\n"
        return useful_information

    def _format_success(self, url: str, goal: str, evidence: str, summary: str) -> str:
        useful_information = f"The useful information in {url} for user goal {goal or 'N/A'} as follows: \n\n"
        useful_information += "Evidence in page: \n" + evidence + "\n\n"
        useful_information += "Summary: \n" + (summary or "No summary generated") + "\n\n"
        return useful_information

