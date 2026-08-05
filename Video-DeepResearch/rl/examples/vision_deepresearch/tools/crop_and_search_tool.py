"""
Crop and search tool for Vision-DeepResearch multi-turn tool-calling rollout.

Based on vision_deepresearch_async_workflow/tools/crop_and_search_tool.py
"""

import asyncio
import base64
import hashlib
import io
import json
import os
import random
import re
from typing import Any, Dict, List, Optional, Tuple, Union

from examples.vision_deepresearch.tools.shared import (
    DeepResearchTool,
    call_extract_model_async,
    log_tool_event,
    parse_json_from_model_output,
    run_with_retries_async,
)


# Try to import optional dependencies for crop_and_search tool
try:
    from PIL import Image
    import requests
    import oss2

    oss2.defaults.connection_pool_size = 10240
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


class CropAndSearchTool(DeepResearchTool):
    """Crop and search tool for visual deep research."""

    MAX_URLS = 3

    def __init__(self):
        if not PIL_AVAILABLE:
            raise ImportError(
                "CropAndSearchTool requires PIL, requests, and oss2 packages"
            )

        super().__init__(
            name="crop_and_search",
            description="Crop regions from an image and perform visual search to gather information. Takes an image_id (path or URL), bbox coordinates (single or multiple), and goal description.",
            parameters={
                "type": "object",
                "properties": {
                    "image_id": {
                        "type": "string",
                        "description": "Path or URL of the image to process",
                    },
                    "bbox": {
                        "type": "array",
                        "items": {
                            "anyOf": [
                                {
                                    "type": "array",
                                    "items": {"type": "number"},
                                    "minItems": 4,
                                    "maxItems": 4,
                                },
                                {"type": "number"},
                            ]
                        },
                        "description": "Bounding box coordinates [x1,y1,x2,y2] or array of bboxes",
                    },
                    "goal": {
                        "type": "string",
                        "description": "Description of what to search for in the cropped regions",
                    },
                },
                "required": ["image_id", "bbox"],
            },
        )
        self.oss_access_key_id = os.getenv("OSS_ACCESS_KEY_ID")
        self.oss_access_key_secret = os.getenv("OSS_ACCESS_KEY_SECRET")
        self.oss_endpoint = os.getenv("OSS_ENDPOINT")
        self.oss_bucket_name = os.getenv("OSS_BUCKET_NAME")
        self.zhipu_api_key = os.getenv("ZHIPU_API_KEY")
        self.jina_api_key = os.getenv("JINA_API_KEY")
        self.serp_api_key = os.getenv("SERP_API_KEY")
        self.zhipu_image_search_url = os.getenv(
            "IMAGE_SEARCH_URL",
            "https://search-svip.bigmodel.cn/api/paas/v4/image_search",
        )
        self.serp_image_search_url = os.getenv(
            "IMAGE_SEARCH_URL",
            "https://google.serper.dev/lens",
        )
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
        self.image_crop_cache = os.getenv("IMAGE_CROP_CACHE", None)
        self.location = os.getenv("SEARCH_LOCATION", "us")
        self._oss_bucket = None

    def _get_oss_bucket(self):
        """Get or create OSS bucket instance."""
        if self._oss_bucket is None:
            self._oss_bucket = oss2.Bucket(
                oss2.Auth(self.oss_access_key_id, self.oss_access_key_secret),
                self.oss_endpoint,
                self.oss_bucket_name,
            )
        return self._oss_bucket

    def _select_extract_url(self) -> str | None:
        if not self.extract_urls:
            return None
        return random.choice(self.extract_urls)

    def _crop_image_by_bbox(
        self, image_path: str, bbox: List[int], output_dir: str
    ) -> Optional[str]:
        """Crop image by bounding box coordinates."""
        try:
            os.makedirs(output_dir, exist_ok=True)

            with Image.open(image_path) as img:
                if img.mode != "RGB":
                    img = img.convert("RGB")

                width, height = img.size

                # Convert coordinates (assuming bbox is in 0-1000 range)
                x1 = max(0, min(int(bbox[0] * width / 1000), width - 1))
                y1 = max(0, min(int(bbox[1] * height / 1000), height - 1))
                x2 = max(0, min(int(bbox[2] * width / 1000), width - 1))
                y2 = max(0, min(int(bbox[3] * height / 1000), height - 1))

                if x2 <= x1 or y2 <= y1:
                    log_tool_event(
                        "CropAndSearch", "InvalidBbox", f"bbox={bbox}", level="WARNING"
                    )
                    return None

                # Crop and resize
                cropped_img = img.crop((x1, y1, x2, y2))
                cropped_img = cropped_img.resize(
                    (cropped_img.width * 2, cropped_img.height * 2),
                    Image.Resampling.LANCZOS,
                )

                # Content-hash filename: identical crops dedup to a single file
                # (avoids collisions across videos that share the same frame basename,
                # and reuses prior identical crops without rewriting).
                buf = io.BytesIO()
                cropped_img.save(buf, "JPEG", quality=95)
                img_bytes = buf.getvalue()
                content_hash = hashlib.md5(img_bytes).hexdigest()
                output_path = os.path.join(output_dir, f"crop_{content_hash}.jpg")

                if not os.path.exists(output_path):
                    # Atomic-ish write: temp file then rename
                    tmp_path = f"{output_path}.tmp.{os.getpid()}"
                    with open(tmp_path, "wb") as fout:
                        fout.write(img_bytes)
                    os.replace(tmp_path, output_path)

                return output_path

        except Exception as e:
            log_tool_event("CropAndSearch", "CropError", str(e), level="ERROR")
            return None

    def _upload_to_oss(self, local_path: str) -> Optional[str]:
        """Upload local image to OSS."""
        try:
            filename = os.path.basename(local_path)
            oss_path = filename

            bucket = self._get_oss_bucket()
            with open(local_path, "rb") as f:
                bucket.put_object(oss_path, f)

            endpoint_host = self.oss_endpoint.replace("https://", "").replace(
                "http://", ""
            )
            public_url = f"https://{self.oss_bucket_name}.{endpoint_host}/{oss_path}"
            return public_url

        except Exception as e:
            log_tool_event("CropAndSearch", "UploadError", str(e), level="ERROR")
            return None

    async def _image_search(self, oss_url: str) -> Optional[List[Dict[str, str]]]:
        """Perform image search using Zhipu or Serp API."""
        if self.zhipu_api_key:
            return await self._image_search_with_zhipu(oss_url)
        else:
            return await self._image_search_with_serp(oss_url)

    async def _image_search_with_zhipu(
        self, oss_url: str
    ) -> Optional[List[Dict[str, str]]]:
        headers = {
            "Authorization": self.zhipu_api_key,
            "Content-Type": "application/json",
            "Accept": "*/*",
        }
        payload = {"url": oss_url, "location": self.location}
        proxies = self._get_requests_proxies()

        def make_search_request():
            response = requests.post(
                self.zhipu_image_search_url,
                headers=headers,
                json=payload,
                timeout=30,
                proxies=proxies,
            )
            response.raise_for_status()
            return response

        try:
            response = await run_with_retries_async(
                func=make_search_request,
                executor=self.executor,
            )

            result_data = response.json()
            search_results = result_data.get("search_result", [])

            formatted_results = []
            for item in search_results[: self.MAX_URLS]:
                title = item.get("title", "Untitled")
                image_url = item.get("image_url", "")
                link = item.get("link", "")
                source = item.get("source", "")
                thumbnail_url = item.get("thumbnail_url", "")

                if image_url and link:
                    formatted_results.append(
                        {
                            "title": title,
                            "image_url": image_url,
                            "link": link,
                            "bbox_image_url": oss_url,
                            "source": source,
                            "thumbnail_url": thumbnail_url,
                        }
                    )

            return formatted_results if formatted_results else None

        except Exception as e:
            log_tool_event(
                "CropAndSearch",
                "SearchError",
                f"provider=zhipu url={oss_url} error={str(e)}",
                level="ERROR",
            )
            return None

    async def _image_search_with_serp(
        self, oss_url: str
    ) -> Optional[List[Dict[str, str]]]:
        headers = {
            "X-API-KEY": self.serp_api_key,
            "Content-Type": "application/json",
        }
        payload = {"url": oss_url, "gl": self.location, "hl": "en"}
        proxies = self._get_requests_proxies()

        def make_search_request():
            response = requests.post(
                self.serp_image_search_url,
                headers=headers,
                json=payload,
                timeout=30,
                proxies=proxies,
            )
            response.raise_for_status()
            return response

        try:
            response = await run_with_retries_async(
                func=make_search_request,
                executor=self.executor,
            )

            result_data = response.json()
            search_results = result_data.get("organic", [])

            formatted_results = []
            for item in search_results[: self.MAX_URLS]:
                title = item.get("title", "Untitled")
                image_url = item.get("imageUrl", "")
                link = item.get("link", "")
                source = item.get("source", "")
                thumbnail_url = item.get("thumbnailUrl", "")

                if image_url and link:
                    formatted_results.append(
                        {
                            "title": title,
                            "image_url": image_url,
                            "link": link,
                            "bbox_image_url": oss_url,
                            "source": source,
                            "thumbnail_url": thumbnail_url,
                        }
                    )

            return formatted_results if formatted_results else None

        except Exception as e:
            log_tool_event(
                "CropAndSearch",
                "SearchError",
                f"provider=serp url={oss_url} error={str(e)}",
                level="ERROR",
            )
            return None

    @staticmethod
    def get_num_bytes(base64_str):
        """Calculate bytes from base64 string."""
        padding = 4 - len(base64_str) % 4
        if padding < 4:
            base64_str += "=" * padding
        decoded_bytes = base64.b64decode(base64_str)
        return len(decoded_bytes)

    def _encode_local_file_to_base64(self, file_path: str) -> Optional[str]:
        """Encode a local image file to base64 format."""
        try:
            if not os.path.exists(file_path):
                return None

            with open(file_path, "rb") as image_file:
                extension = file_path.split(".")[-1].lower()
                if extension in ["jpg", "jpeg"]:
                    image_format = "jpeg"
                elif extension == "png":
                    image_format = "png"
                elif extension == "gif":
                    image_format = "gif"
                elif extension == "webp":
                    image_format = "webp"
                elif extension == "bmp":
                    image_format = "bmp"
                else:
                    image_format = "jpeg"

                encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
                return f"data:image/{image_format};base64,{encoded_string}"
        except Exception:
            return None

    def _validate_base64(self, base64_string: str) -> bool:
        """Validate if a base64 string is valid."""
        try:
            if base64_string.startswith("data:image/"):
                if (
                    ";base64," in base64_string
                    and self.get_num_bytes(base64_string) > 15000
                ):
                    base64_part = base64_string.split(";base64,", 1)[1]
                else:
                    return False
            else:
                base64_part = base64_string

            base64.b64decode(base64_part, validate=True)
            return True
        except Exception:
            return False

    def _encode_url_to_base64(self, url: str, timeout: int = 30) -> Optional[str]:
        """Encode a network image URL to base64 format."""
        try:
            proxies = self._get_requests_proxies()

            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                "Accept": "image/webp,image/apng,image/*,*/*;q=0.8",
            }

            response = requests.get(
                url, timeout=timeout, proxies=proxies, headers=headers, stream=True
            )
            response.raise_for_status()

            content = b""
            max_size = 10 * 1024 * 1024  # 10 MB limit
            for chunk in response.iter_content(chunk_size=8192):
                content += chunk
                if len(content) > max_size:
                    return None

            if len(content) == 0:
                return None

            content_type = response.headers.get("content-type", "")
            image_format = "jpeg"

            if content_type.startswith("image/"):
                image_format = content_type.split("/")[-1].split(";")[0].lower()
            else:
                if content[:3] == b"\xff\xd8\xff":
                    image_format = "jpeg"
                elif content[:8] == b"\x89PNG\r\n\x1a\n":
                    image_format = "png"
                elif content[:6] in [b"GIF87a", b"GIF89a"]:
                    image_format = "gif"
                elif content[:4] == b"RIFF" and content[8:12] == b"WEBP":
                    image_format = "webp"

            if image_format not in ["jpeg", "png", "gif", "webp", "bmp"]:
                image_format = "jpeg"

            encoded_string = base64.b64encode(content).decode("utf-8")
            return f"data:image/{image_format};base64,{encoded_string}"

        except Exception:
            return None

    async def _safe_encode_image_to_base64(
        self, image_path: str, timeout: int = 5
    ) -> Optional[str]:
        """Safely encode an image to base64 with validation."""
        try:
            if image_path.startswith(("http://", "https://")):
                result = await self._run_blocking(
                    lambda: self._encode_url_to_base64(image_path, timeout)
                )
            else:
                result = await self._run_blocking(
                    lambda: self._encode_local_file_to_base64(image_path)
                )

            if result and await self._run_blocking(
                lambda: self._validate_base64(result)
            ):
                return result
            return None
        except Exception:
            return None

    def _extract_images_from_content(self, content: str) -> List[Tuple[str, str]]:
        """Extract all image alt texts and URLs from webpage content."""
        pattern = r"!\[(.*?)\]\((https?://[^\s]+)\)"
        matches = re.findall(pattern, content)
        return matches

    async def _summarize_with_extract_only_text(
        self,
        content: str,
        goal: str,
    ) -> Optional[Dict[str, Any]]:
        """Text-only version of webpage content summarization."""
        TEXT_ONLY_PROMPT = """You are a text analysis assistant. Inputs: webpage text and a user's goal. Extract only what helps the goal.

Output ONLY a JSON object (no markdown, no extra text) with three string keys. BE TERSE:
- "rational": ≤ 1 short clause (max ~15 words).
- "evidence": ≤ 2 short sentences, copy only the minimum text needed from the page (no full paragraphs, no padding).
- "summary": ≤ 2 short sentences answering the goal.
Total output must stay under ~120 words. Use empty strings when nothing is available. No other keys.

Example: {"rational": "...", "evidence": "...", "summary": "..."}
"""

        if not content or not content.strip():
            return {
                "rational": "No valid text content extracted from webpage",
                "evidence": "",
                "summary": "Unable to process webpage content, text content is empty",
            }

        max_text_length = 50000
        truncated_content = (
            content[:max_text_length] + "...\n[Content truncated]"
            if len(content) > max_text_length
            else content
        )

        message_content = [
            {"type": "text", "text": f"Webpage content:\n\n{truncated_content}"}
        ]

        if goal:
            message_content.append({"type": "text", "text": f"\nUser's goal: {goal}"})

        messages = [
            {"role": "system", "content": TEXT_ONLY_PROMPT},
            {"role": "user", "content": message_content},
        ]

        extract_url = self._select_extract_url()
        if extract_url:
            messages = [
                {"role": "system", "content": TEXT_ONLY_PROMPT},
                {"role": "user", "content": message_content},
            ]
            raw_payload = await call_extract_model_async(
                url=extract_url,
                model=self.extract_model,
                messages=messages,
                max_tokens=self.extract_max_tokens,
                proxies=self._get_requests_proxies(),
                executor=self.executor,
                source="CropAndSearch/ExtractText",
            )
            parsed = parse_json_from_model_output(raw_payload or "")
            if parsed is not None:
                return parsed

        return None

    async def _summarize_with_extract(
        self,
        content: str,
        goal: str,
        reader_payload: Dict[str, Any],
        query_image_url: Optional[str] = None,
        title: str = "",
        image_url: str = "",
        thumbnail_url: str = "",
        source: str = "",
        max_images: int = 10,
    ) -> Optional[Dict[str, Any]]:
        """Summarize webpage content using visual language model."""
        EXTRACTOR_PROMPT = """You are a multimodal assistant. Inputs: a query image, a goal, and web content (preview metadata/images + page body text/images).

Steps:
1. Compare the query image with the web images and judge whether they show the same entity.
2. If same entity: extract the goal-relevant info. If not: note the key visual difference, then still extract any info useful for the goal.

Output ONLY a JSON object (no markdown, no extra text) with three string keys. BE TERSE:
- "rational": ≤ 1 short clause (max ~15 words).
- "evidence": ≤ 2 short sentences, copy only the minimum text needed from the page (no full paragraphs, no padding).
- "summary": ≤ 2 short sentences answering the goal.
Total output must stay under ~120 words. Use empty strings when nothing is available. No other keys.

Example: {"rational": "...", "evidence": "...", "summary": "..."}
"""

        message_content: List[Dict[str, Any]] = []

        # 1. User's query image
        if query_image_url:
            query_image_base64 = await self._safe_encode_image_to_base64(query_image_url)
            if query_image_base64:
                message_content.append(
                    {
                        "type": "text",
                        "text": "User's query image (the image the user is searching for):",
                    }
                )
                message_content.append(
                    {"type": "image_url", "image_url": {"url": query_image_base64}}
                )

        # 2. User's goal
        if goal:
            message_content.append({"type": "text", "text": f"User's goal:\n{goal}"})

        # 3. Search result metadata
        preview_parts = []
        if source:
            preview_parts.append(f"Website source: {source}")
        if title:
            preview_parts.append(f"Page title: {title}")

        if preview_parts:
            message_content.append(
                {
                    "type": "text",
                    "text": "Search result metadata:\n" + "\n".join(preview_parts),
                }
            )

        # Preview images
        preview_items = [
            (label, url)
            for label, url in [("Main image", image_url), ("Thumbnail", thumbnail_url)]
            if url
        ]
        if preview_items:
            preview_tasks = [
                self._safe_encode_image_to_base64(url) for _, url in preview_items
            ]
            preview_results = await asyncio.gather(*preview_tasks)
            for (label, _), img_b64 in zip(preview_items, preview_results):
                if img_b64:
                    message_content.append(
                        {"type": "text", "text": f"Search result {label}:"}
                    )
                    message_content.append(
                        {"type": "image_url", "image_url": {"url": img_b64}}
                    )

        # 4. Webpage content
        if content.strip():
            max_text_length = 50000
            truncated_content = (
                content[:max_text_length] + "...\n[Content truncated]"
                if len(content) > max_text_length
                else content
            )
            message_content.append(
                {"type": "text", "text": "Webpage content:\n\n" + truncated_content}
            )

        # 5. Images from webpage content
        if content:
            image_matches = self._extract_images_from_content(content)
            if image_matches:
                selected_matches = image_matches[:max_images]
                image_tasks = [
                    self._safe_encode_image_to_base64(img_url)
                    for _, img_url in selected_matches
                ]
                image_results = await asyncio.gather(*image_tasks)
                webpage_images = [
                    (alt_text, img_base64)
                    for (alt_text, _), img_base64 in zip(selected_matches, image_results)
                    if img_base64
                ]

                if webpage_images:
                    message_content.append(
                        {"type": "text", "text": "Images from webpage:"}
                    )
                    for alt_text, img_base64 in webpage_images:
                        if alt_text.strip():
                            message_content.append(
                                {"type": "text", "text": f"Image '{alt_text}':"}
                            )
                        message_content.append(
                            {"type": "image_url", "image_url": {"url": img_base64}}
                        )

        # Check if we have content
        has_content = any(
            (item["type"] == "text" and item["text"].strip())
            or item["type"] == "image_url"
            for item in message_content
        )

        if not has_content:
            return {
                "rational": "No valid content extracted from webpage or search results.",
                "evidence": "",
                "summary": "Unable to process webpage and search preview content.",
            }

        # Try extract service
        extract_url = self._select_extract_url()
        if extract_url:
            messages = [
                {"role": "system", "content": EXTRACTOR_PROMPT},
                {"role": "user", "content": message_content},
            ]
            raw_payload = await call_extract_model_async(
                url=extract_url,
                model=self.extract_model,
                messages=messages,
                max_tokens=self.extract_max_tokens,
                proxies=self._get_requests_proxies(),
                executor=self.executor,
                source="CropAndSearch/Extract",
            )
            parsed = parse_json_from_model_output(raw_payload or "")
            if parsed is not None:
                return parsed

        # Fallback to text-only
        return await self._summarize_with_extract_only_text(content, goal)

    async def _fetch_reader_content(self, url: str) -> Optional[Dict[str, Any]]:
        """Fetch webpage content using Reader API."""
        try:
            proxies = self._get_requests_proxies()

            if self.zhipu_api_key:
                headers = {"Content-Type": "application/json"}
                headers["Authorization"] = self.zhipu_api_key

                optional_headers = {
                    "X-Return-Format": "markdown",
                    "X-No-Cache": "false",
                    "X-Timeout": "30",
                    "X-Retain-Images": "true",
                    "X-With-Images-Summary": "true",
                    "X-With-Links-Summary": "true",
                }
                headers.update(
                    {k: v for k, v in optional_headers.items() if v is not None}
                )

                body = {"url": url}

                def send_request():
                    return requests.post(
                        self.zhipu_reader_url,
                        headers=headers,
                        json=body,
                        timeout=30,
                        proxies=proxies,
                    )

                response = await run_with_retries_async(
                    send_request, executor=self.executor
                )

                if response.status_code != 200:
                    log_tool_event(
                        "CropAndSearch",
                        "ReaderBadStatus",
                        (
                            f"provider=zhipu url={url} reader_url={self.zhipu_reader_url} "
                            f"status={response.status_code} body={response.text[:1000]}"
                        ),
                        level="ERROR",
                    )
                    return None

                try:
                    payload = response.json()
                except json.JSONDecodeError as exc:
                    log_tool_event(
                        "CropAndSearch",
                        "ReaderNonJsonPayload",
                        (
                            f"provider=zhipu url={url} reader_url={self.zhipu_reader_url} "
                            f"body={response.text[:1000]}"
                        ),
                        error=str(exc),
                        level="ERROR",
                    )
                    return None

                if isinstance(payload.get("reader_result"), dict):
                    data = payload["reader_result"]
                    payload_format = "reader_result"
                elif payload.get("code") == 200 and isinstance(payload.get("data"), dict):
                    data = payload["data"]
                    payload_format = "code_data"
                else:
                    log_tool_event(
                        "CropAndSearch",
                        "ReaderUnexpectedPayload",
                        (
                            f"provider=zhipu url={url} keys={list(payload.keys())} "
                            f"code={payload.get('code')} "
                            f"message={payload.get('msg') or payload.get('message') or payload.get('error')}"
                        ),
                        level="ERROR",
                    )
                    return None

                meta = dict(data)
                meta.setdefault("provider", "zhipu")
                meta.setdefault("payload_format", payload_format)
                meta.setdefault("reader_url", self.zhipu_reader_url)
                for key in ("id", "request_id", "model"):
                    if payload.get(key) is not None:
                        meta.setdefault(key, payload[key])

                result = {
                    "content": data.get("content") or "",
                    "description": data.get("description") or "",
                    "meta": meta,
                }
            else:
                headers = {"Authorization": self.jina_api_key}
                body = {"url": url}

                def send_request():
                    return requests.post(
                        self.jina_reader_url,
                        headers=headers,
                        data=body,
                        timeout=30,
                        proxies=proxies,
                    )

                response = await run_with_retries_async(
                    send_request, executor=self.executor
                )

                if response.status_code != 200:
                    return None

                result = {
                    "content": response.text or "",
                    "description": "",
                    "meta": {
                        "provider": "jina",
                        "url": url,
                    },
                }

            return result

        except Exception:
            return None

    async def _visit_webpages_for_search(
        self, search_results: List[Dict[str, str]], goal: str
    ) -> str:
        """Visit webpages for search results and extract relevant information."""
        try:
            # Create concurrent tasks for all webpage visits
            visit_tasks = [
                self._handle_single_url(
                    url=item["link"],
                    goal=goal,
                    query_image_url=item["bbox_image_url"],
                    title=item["title"],
                    thumbnail_url=item["thumbnail_url"],
                    image_url=item["image_url"],
                    source=item["source"],
                )
                for item in search_results
            ]

            # Execute all webpage visits concurrently
            visit_results = await asyncio.gather(*visit_tasks, return_exceptions=True)

            # Process results
            all_results = []
            for i, result in enumerate(visit_results):
                try:
                    if isinstance(result, Exception):
                        log_tool_event(
                            "CropAndSearch",
                            "VisitTaskException",
                            f"webpage_{i+1} error={str(result)}",
                            level="ERROR",
                        )
                        all_results.append(
                            f"[{i+1}] [Error visiting webpage: {str(result)}]"
                        )
                    elif isinstance(result, str):
                        all_results.append(f"[{i+1}] {result}")
                    else:
                        log_tool_event(
                            "CropAndSearch",
                            "InvalidVisitResult",
                            f"webpage_{i+1} unexpected_result_type={type(result)}",
                            level="ERROR",
                        )
                        all_results.append(
                            f"[{i+1}] [Invalid result format: {type(result)}]"
                        )
                except Exception as e:
                    log_tool_event(
                        "CropAndSearch",
                        "VisitResultProcessingError",
                        f"webpage_{i+1} error={str(e)}",
                        level="ERROR",
                    )
                    all_results.append(
                        f"[{i+1}] [Error processing visit result: {str(e)}]"
                    )

            return "\n\n=======\n\n".join(all_results)

        except Exception as e:
            log_tool_event("CropAndSearch", "VisitSetupError", str(e), level="ERROR")
            return f"[Error setting up webpage visits: {str(e)}]"

    async def _handle_single_url(
        self,
        url: str,
        goal: str,
        query_image_url: Optional[str] = None,
        title: str = "",
        thumbnail_url: str = "",
        image_url: str = "",
        source: str = "",
        max_content_chars: int = 120000,
    ) -> str:
        """Handle visiting a single URL."""
        try:
            reader_payload = await self._fetch_reader_content(url)
            if not reader_payload:
                log_tool_event(
                    "CropAndSearch",
                    "ReaderFetchFailed",
                    f"url={url} title={title}",
                    level="ERROR",
                )
                return f"[Error] Failed to fetch content from [{title}]({url})"

            content = reader_payload.get("content") or ""
            description = reader_payload.get("description") or ""

            if not content:
                content = "Webpage content is empty."

            # Truncate content for display
            if len(content) > max_content_chars:
                content = content[:max_content_chars] + "\n[Content truncated...]"

            # Try visual summarization with extract service
            summary_result = await self._summarize_with_extract(
                content=content,
                goal=goal,
                reader_payload=reader_payload,
                query_image_url=query_image_url,
                title=title,
                image_url=image_url,
                thumbnail_url=thumbnail_url,
                source=source,
            )

            if summary_result:
                rational_text = summary_result.get("rational") or ""
                evidence_text = summary_result.get("evidence") or content[:2000] + (
                    "..." if len(content) > 2000 else ""
                )
                summary_text = summary_result.get("summary") or description or ""
            else:
                log_tool_event(
                    "CropAndSearch",
                    "ExtractSummaryFailed",
                    f"url={url} title={title}",
                    level="ERROR",
                )
                rational_text = ""
                evidence_text = content[:2000] + ("..." if len(content) > 2000 else "")
                summary_text = description or "Summary unavailable."

            result = f"The useful information in [{title}]({url}) are:\n\n"
            result += f"Evidence in page:\n{evidence_text}\n\n"
            result += f"Summary:\n{summary_text}\n\n"

            return result

        except Exception as e:
            return f"[Error] Failed to process {url}: {str(e)}"

    async def _process_single_bbox(
        self, bbox: List[int], bbox_index: int, image_id: str, cache_dir: str, goal: str
    ) -> Tuple[int, str, Optional[str]]:
        """Process a single bounding box concurrently."""
        try:
            # 1. Crop image
            cropped_path = await self._run_blocking(
                lambda: self._crop_image_by_bbox(image_id, bbox, cache_dir)
            )
            if not cropped_path:
                log_tool_event(
                    "CropAndSearch",
                    "CropFailed",
                    f"bbox={bbox} image_id={image_id}",
                    level="ERROR",
                )
                return bbox_index, f"Bbox {bbox}: Image cropping failed", None

            # 2. Upload to OSS
            oss_url = await self._run_blocking(
                lambda: self._upload_to_oss(cropped_path)
            )
            if not oss_url:
                log_tool_event(
                    "CropAndSearch",
                    "UploadFailed",
                    f"bbox={bbox} cropped_path={cropped_path}",
                    level="ERROR",
                )
                return bbox_index, f"Bbox {bbox}: OSS upload failed", None

            # 3. Perform image search
            search_results = await self._image_search(oss_url)
            if not search_results:
                log_tool_event(
                    "CropAndSearch",
                    "ImageSearchFailed",
                    f"bbox={bbox} oss_url={oss_url}",
                    level="ERROR",
                )
                return bbox_index, f"Bbox {bbox}: Image search failed", oss_url

            # 4. Visit webpages
            visit_results = await self._visit_webpages_for_search(search_results, goal)

            result_text = (
                f"The search results for bbox {bbox} are as follows:\n{visit_results}"
            )
            return bbox_index, result_text, oss_url

        except Exception as e:
            log_tool_event(
                "CropAndSearch",
                "BboxError",
                f"bbox_{bbox_index+1} error={str(e)}",
                level="ERROR",
            )
            return bbox_index, f"Bbox {bbox}: Processing failed - {str(e)}", None

    async def call(
        self,
        image_id: str,
        bbox: Union[List[int], List[List[int]]],
        goal: str = "",
        **kwargs,
    ) -> str:
        """Execute crop and search operation."""
        cache_dir = self.image_crop_cache
        if cache_dir is None:
            return "[CropAndSearch] IMAGE_CROP_CACHE must be provided."
        os.makedirs(cache_dir, exist_ok=True)

        try:
            # Normalize bbox format
            if isinstance(bbox, list) and len(bbox) > 0:
                if isinstance(bbox[0], list):
                    bbox_list = bbox
                else:
                    bbox_list = [bbox]
            else:
                return "[CropAndSearch] Invalid bbox format"

            # Create concurrent tasks for all bboxes
            tasks = [
                self._process_single_bbox(single_bbox, i, image_id, cache_dir, goal)
                for i, single_bbox in enumerate(bbox_list)
            ]

            # Execute all tasks concurrently
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Process results and maintain order
            all_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    log_tool_event(
                        "CropAndSearch",
                        "TaskException",
                        f"bbox_{i+1} error={str(result)}",
                        level="ERROR",
                    )
                    all_results.append(f"Bbox {bbox_list[i]}: Task failed - {str(result)}")
                elif isinstance(result, tuple) and len(result) >= 2:
                    _, result_text, _ = result
                    all_results.append(result_text)
                else:
                    log_tool_event(
                        "CropAndSearch",
                        "InvalidResult",
                        f"bbox_{i+1} unexpected_result_type={type(result)}",
                        level="ERROR",
                    )
                    all_results.append(f"Bbox {bbox_list[i]}: Invalid result format")

            return "\n\n=======\n\n".join(all_results)

        except Exception as e:
            log_tool_event("CropAndSearch", "ExecutionError", str(e), level="ERROR")
            return f"[CropAndSearch Error] {str(e)}"


