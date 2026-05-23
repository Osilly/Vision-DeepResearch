"""
DeepResearch Tools - Production-ready implementations

This module provides tool implementations for the DeepResearch agent, with real
functionality ported from Tongyi's original implementations where possible.

Now supports both:
- ReAct text format (for gpt-4o, Claude, etc.)
- OpenAI native function calling (for o3, o3-mini, etc.)
"""

import asyncio
import base64
import hashlib
import http.client
import json
import os
import re
import sqlite3
import threading
import time
import uuid
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any, TypeVar, List, Dict, Union, Tuple, Optional

# --- Lightweight RLLMTool stub (drops the rllm dependency) ------------------- #
# The reference implementation inherits from `rllm.tools.tool_base.Tool`. This
# standalone eval only needs a minimal base class that supports the
# `(name, description, parameters)` constructor signature used throughout the
# tools, so we provide it locally.
class RLLMTool(ABC):
    def __init__(self, name: str = "", description: str = "", parameters: dict | None = None):
        self.name = name
        self.description = description
        self.parameters = parameters or {}

    @abstractmethod
    def call(self, *args, **kwargs):  # noqa: D401 - matches reference signature
        ...


# Try to import optional dependencies for crop_and_search tool.
# Import-time failures must NOT break `from tools import get_all_tools` —
# tools that need these deps will raise at .call() time instead.
try:
    from PIL import Image  # noqa: F401
    import requests  # noqa: F401
    import oss2
    oss2.defaults.connection_pool_size = 512
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    Image = None  # type: ignore
    requests = None  # type: ignore
    oss2 = None  # type: ignore


T = TypeVar("T")
def _normalize_level(level: str | None) -> str:
    if not level:
        return "INFO"
    return str(level).upper()


def run_with_retries(func: Callable[[], T], attempts: int = 5, delay: float = 1.0) -> T:
    """Execute a callable with retry support."""

    last_error: Exception | None = None
    for attempt in range(1, max(attempts, 1) + 1):
        try:
            return func()
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt >= attempts:
                break
            if delay > 0:
                time.sleep(delay)

    if last_error is not None:
        raise last_error

    raise RuntimeError("run_with_retries executed without performing any attempts")


def shorten_for_log(text: str, limit: int = 200) -> str:
    """Create a concise preview string for debug logging."""

    if text is None:
        return ""

    if not isinstance(text, str):
        text = str(text)

    if not text:
        return ""

    normalized = text.replace("\n", "\\n")
    if len(normalized) <= limit * 2:
        return normalized
    return f"{normalized[:limit]} ... {normalized[-limit:]}"


# Configuration for crop and search tool
OSS_CONFIG = {
    "access_key_id": os.getenv("OSS_ACCESS_KEY_ID", ""),
    "access_key_secret": os.getenv("OSS_ACCESS_KEY_SECRET", ""),
    "endpoint": os.getenv("OSS_ENDPOINT", "https://oss-cn-shanghai.aliyuncs.com"),
    "bucket_name": os.getenv("OSS_BUCKET_NAME", ""),
}

SEARCH_CONFIG = {
    "zhipu_key": os.getenv("ZHIPU_API_KEY", ""),
    "search_url": os.getenv("SEARCH_URL", "https://search-svip.bigmodel.cn/api/paas/v4/image_search"),
}

# Global OSS bucket instance
_oss_bucket = None

# Cache database configuration
CACHE_CONFIG = {
    "db_path": os.getenv("CACHE_DB_PATH", "deepresearch_cache.db"),
    "max_age_days": int(os.getenv("CACHE_MAX_AGE_DAYS", "30")),  # Cache validity period
    "max_retries": int(os.getenv("CACHE_MAX_RETRIES", "3")),  # Max retry attempts
    "base_retry_delay": float(os.getenv("CACHE_RETRY_DELAY", "0.1")),  # Base delay in seconds
    "busy_timeout": int(os.getenv("CACHE_BUSY_TIMEOUT", "30000")),  # SQLite busy timeout in ms
}

# Thread-local storage for database connections
_local = threading.local()


def _create_cache_tables():
    """Create cache tables if they don't exist."""
    try:
        db = get_cache_db()
        for table_sql in CACHE_TABLES.values():
            db.executescript(table_sql)
        db.commit()
        log_tool_event("Cache", "TablesCreated", f"Created {len(CACHE_TABLES)} cache tables")
    except Exception as e:
        log_tool_event("Cache", "TableCreateError", f"Failed to create tables: {str(e)}", level="ERROR")


# Cache table schemas
CACHE_TABLES = {
    "text_search": """
        CREATE TABLE IF NOT EXISTS text_search (
            query_hash TEXT PRIMARY KEY,
            query TEXT NOT NULL,
            result TEXT NOT NULL,
            created_at REAL NOT NULL,
            last_accessed REAL NOT NULL,
            access_count INTEGER DEFAULT 1
        );
        CREATE INDEX IF NOT EXISTS idx_text_search_query_hash ON text_search(query_hash);
        CREATE INDEX IF NOT EXISTS idx_text_search_last_accessed ON text_search(last_accessed);
    """,
    "text_visit": """
        CREATE TABLE IF NOT EXISTS text_visit (
            url_hash TEXT PRIMARY KEY,
            url TEXT NOT NULL,
            result TEXT NOT NULL,
            created_at REAL NOT NULL,
            last_accessed REAL NOT NULL,
            access_count INTEGER DEFAULT 1
        );
        CREATE INDEX IF NOT EXISTS idx_text_visit_url_hash ON text_visit(url_hash);
        CREATE INDEX IF NOT EXISTS idx_text_visit_last_accessed ON text_visit(last_accessed);
    """,
    "image_search": """
        CREATE TABLE IF NOT EXISTS image_search (
            image_url_hash TEXT PRIMARY KEY,
            image_url TEXT NOT NULL,
            result TEXT NOT NULL,
            created_at REAL NOT NULL,
            last_accessed REAL NOT NULL,
            access_count INTEGER DEFAULT 1
        );
        CREATE INDEX IF NOT EXISTS idx_image_search_url_hash ON image_search(image_url_hash);
        CREATE INDEX IF NOT EXISTS idx_image_search_last_accessed ON image_search(last_accessed);
    """,
    "image_visit": """
        CREATE TABLE IF NOT EXISTS image_visit (
            url_hash TEXT PRIMARY KEY,
            url TEXT NOT NULL,
            result TEXT NOT NULL,
            created_at REAL NOT NULL,
            last_accessed REAL NOT NULL,
            access_count INTEGER DEFAULT 1
        );
        CREATE INDEX IF NOT EXISTS idx_image_visit_url_hash ON image_visit(url_hash);
        CREATE INDEX IF NOT EXISTS idx_image_visit_last_accessed ON image_visit(last_accessed);
    """
}


def get_oss_bucket():
    """Get or create OSS bucket instance."""
    global _oss_bucket
    if _oss_bucket is None:
        _oss_bucket = oss2.Bucket(
            oss2.Auth(OSS_CONFIG["access_key_id"], OSS_CONFIG["access_key_secret"]),
            OSS_CONFIG["endpoint"],
            OSS_CONFIG["bucket_name"]
        )
    return _oss_bucket


def get_cache_db():
    """Get thread-local database connection with enhanced error handling."""
    if not hasattr(_local, 'db'):
        try:
            # Ensure database directory exists
            db_dir = os.path.dirname(CACHE_CONFIG["db_path"])
            if db_dir and not os.path.exists(db_dir):
                os.makedirs(db_dir, exist_ok=True)

            _local.db = sqlite3.connect(CACHE_CONFIG["db_path"], check_same_thread=False)
            _local.db.execute("PRAGMA journal_mode=WAL")  # Enable WAL mode for better concurrency
            _local.db.execute("PRAGMA synchronous=NORMAL")  # Balance performance and safety
            _local.db.execute("PRAGMA cache_size=-64000")  # 64MB cache
            _local.db.execute(f"PRAGMA busy_timeout={CACHE_CONFIG['busy_timeout']}")  # Configurable timeout
            _local.db.execute("PRAGMA wal_autocheckpoint=1000")  # Auto checkpoint WAL
            _local.db.execute("PRAGMA foreign_keys=ON")  # Enable foreign key constraints
            _local.db.execute("PRAGMA temp_store=MEMORY")  # Store temp tables in memory

            # Test connection with a simple query
            _local.db.execute("SELECT 1").fetchone()

            log_tool_event("Cache", "Init", f"Database initialized: {CACHE_CONFIG['db_path']}")

        except sqlite3.Error as e:
            log_tool_event("Cache", "InitError", f"Failed to initialize database: {str(e)}", level="ERROR")
            # Fallback: try to create connection without advanced features
            try:
                _local.db = sqlite3.connect(":memory:", check_same_thread=False)
                log_tool_event("Cache", "Fallback", "Using in-memory database as fallback")
            except Exception as fallback_e:
                log_tool_event("Cache", "CriticalError", f"Failed to create any database connection: {str(fallback_e)}", level="CRITICAL")
                raise RuntimeError("Cannot initialize cache database") from fallback_e
        initialize_cache_tables(_local.db)
    return _local.db


def initialize_cache_tables(db: sqlite3.Connection):
    """Initialize cache tables if they don't exist."""
    for table_name, schema in CACHE_TABLES.items():
        db.executescript(schema)
    db.commit()


def cleanup_expired_cache():
    """Clean up expired cache entries."""
    try:
        db = get_cache_db()
        cutoff_time = time.time() - (CACHE_CONFIG["max_age_days"] * 24 * 60 * 60)

        tables = ["text_search", "text_visit", "image_search", "image_visit"]
        for table in tables:
            db.execute(f"DELETE FROM {table} WHERE last_accessed < ?", (cutoff_time,))

        db.commit()
        log_tool_event("Cache", "Cleanup", f"Cleaned expired entries older than {CACHE_CONFIG['max_age_days']} days")
    except Exception as e:
        log_tool_event("Cache", "CleanupError", str(e), level="ERROR")


def get_cache_key(text: str) -> str:
    """Generate a cache key from text using SHA256."""
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def get_cache(table: str, key: str) -> Optional[str]:
    """Get cached result from database with comprehensive error handling."""
    max_retries = CACHE_CONFIG["max_retries"]
    base_delay = CACHE_CONFIG["base_retry_delay"]

    for attempt in range(max_retries):
        try:
            db = get_cache_db()
            current_time = time.time()

            # Define column names for each table
            column_mapping = {
                "text_search": "query_hash",
                "text_visit": "url_hash",
                "image_search": "image_url_hash",
                "image_visit": "url_hash"
            }

            hash_col = column_mapping.get(table, "hash")

            # Use transaction for atomic update and select
            with db:
                # Update last_accessed and access_count, then get result
                db.execute(f"""
                    UPDATE {table} SET last_accessed = ?, access_count = access_count + 1
                    WHERE {hash_col} = ?
                """, (current_time, key))

                cursor = db.execute(f"SELECT result FROM {table} WHERE {hash_col} = ?", (key,))
                row = cursor.fetchone()

                if row:
                    log_tool_event("Cache", "Hit", f"table={table} key={key[:8]}...")
                    return row[0]
                else:
                    log_tool_event("Cache", "Miss", f"table={table} key={key[:8]}...")
                    return None

        except sqlite3.OperationalError as e:
            error_msg = str(e).lower()
            if ("database is locked" in error_msg or "database is busy" in error_msg) and attempt < max_retries - 1:
                # Wait with exponential backoff before retry
                wait_time = (2 ** attempt) * base_delay
                time.sleep(wait_time)
                log_tool_event("Cache", "Retry", f"table={table} attempt={attempt+1}/{max_retries} wait={wait_time:.2f}s error={str(e)[:50]}...")
                continue
            else:
                log_tool_event("Cache", "GetError", f"table={table} error={str(e)}", level="ERROR")
                return None

        except sqlite3.DatabaseError as e:
            # Database corruption or other serious issues
            log_tool_event("Cache", "CorruptionError", f"table={table} error={str(e)}", level="CRITICAL")
            # Try to reinitialize database connection
            if hasattr(_local, 'db'):
                try:
                    _local.db.close()
                except:
                    pass
                delattr(_local, 'db')
            return None

        except Exception as e:
            log_tool_event("Cache", "GetError", f"table={table} error={str(e)}", level="ERROR")
            return None


def set_cache(table: str, key: str, original_input: str, result: str):
    """Store result in cache with comprehensive error handling."""
    max_retries = CACHE_CONFIG["max_retries"]
    base_delay = CACHE_CONFIG["base_retry_delay"]

    for attempt in range(max_retries):
        try:
            db = get_cache_db()
            current_time = time.time()

            # Define column names for each table
            column_mapping = {
                "text_search": ("query_hash", "query"),
                "text_visit": ("url_hash", "url"),
                "image_search": ("image_url_hash", "image_url"),
                "image_visit": ("url_hash", "url")
            }

            hash_col, input_col = column_mapping.get(table, ("hash", "input"))

            # Validate data sizes to prevent database issues
            if len(result) > 100 * 1024 * 1024:  # 100MB limit
                log_tool_event("Cache", "SizeError", f"table={table} result too large: {len(result)} bytes", level="WARNING")
                return

            # Use transaction for atomic operation
            with db:
                # Insert or replace
                db.execute(f"""
                    INSERT OR REPLACE INTO {table}
                    ({hash_col}, {input_col}, result, created_at, last_accessed, access_count)
                    VALUES (?, ?, ?, ?, ?, 1)
                """, (key, original_input, result, current_time, current_time))

                log_tool_event("Cache", "Stored", f"table={table} key={key[:8]}... size={len(result)}")
                return  # Success, exit function

        except sqlite3.OperationalError as e:
            error_msg = str(e).lower()
            if ("database is locked" in error_msg or "database is busy" in error_msg) and attempt < max_retries - 1:
                # Wait with exponential backoff before retry
                wait_time = (2 ** attempt) * base_delay
                time.sleep(wait_time)
                log_tool_event("Cache", "Retry", f"table={table} attempt={attempt+1}/{max_retries} wait={wait_time:.2f}s error={str(e)[:50]}...")
                continue
            else:
                log_tool_event("Cache", "SetError", f"table={table} error={str(e)}", level="ERROR")
                return

        except sqlite3.DatabaseError as e:
            # Database corruption or other serious issues
            log_tool_event("Cache", "CorruptionError", f"table={table} error={str(e)}", level="CRITICAL")
            # Try to reinitialize database connection
            if hasattr(_local, 'db'):
                try:
                    _local.db.close()
                except:
                    pass
                delattr(_local, 'db')
            return

        except Exception as e:
            log_tool_event("Cache", "SetError", f"table={table} error={str(e)}", level="ERROR")
            return


def log_tool_event(source: str, status: str, message: str | None, *, error: str | None = None, level: str | None = "INFO") -> None:
    """Unified logging helper for DeepResearch tools (stdout based)."""

    safe_message = message or ""
    message_preview = shorten_for_log(safe_message)
    level_name = _normalize_level(level)

    log_parts = [
        f"[Tool][{source}][{status}][{level_name}]",
        f"message_len={len(safe_message)}",
        f"preview={json.dumps(message_preview, ensure_ascii=False)}",
    ]

    if error is not None:
        error_preview = shorten_for_log(error)
        log_parts.append(f"error_len={len(error)}")
        log_parts.append(f"error={json.dumps(error_preview, ensure_ascii=False)}")

    print(" ".join(log_parts))


def log_search(source: str, status: str, query: str, result: str | None = None, error: str | None = None) -> None:
    """Standardized debug logs for search tools."""

    parts = [f"query={json.dumps(query, ensure_ascii=False)}"]

    if result is not None:
        preview = shorten_for_log(result)
        parts.append(f"result_len={len(result)}")
        parts.append(f"preview={json.dumps(preview, ensure_ascii=False)}")

    message = " ".join(parts)
    level = "ERROR" if error else "INFO"

    log_tool_event(
        source=f"Search/{source}",
        status=status,
        message=message,
        error=error,
        level=level,
    )


class DeepResearchTool(RLLMTool, ABC):
    """
    Base class for all DeepResearch tools.

    Inherits from rLLM's Tool to support OpenAI native function calling,
    while maintaining compatibility with ReAct text format.
    """

    def __init__(self, name: str, description: str, parameters: dict | None = None):
        """
        Initialize DeepResearch tool with OpenAI function calling support.

        Args:
            name: Tool name
            description: Tool description
            parameters: OpenAI-style parameter schema (optional)
        """
        # Set _json BEFORE calling super().__init__
        # because the parent's __init__ may access self.json
        self._json = {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": parameters or {"type": "object", "properties": {}, "required": []},
            },
        }

        super().__init__(name=name, description=description)

    @abstractmethod
    async def call(self, **kwargs) -> str:
        """Execute the tool with given arguments."""
        pass

    def _get_requests_proxies(self) -> dict | None:
        """Build requests-compatible proxy mapping from TOOL_HTTPS_PROXY."""
        proxy_value = os.getenv("TOOL_HTTPS_PROXY")
        if proxy_value is None:
            return None

        proxy_value = proxy_value.strip()
        if not proxy_value or proxy_value.lower() == "none":
            return {"http": None, "https": None}

        return {"http": proxy_value, "https": proxy_value}

    async def async_forward(self, **kwargs):
        """rLLM Tool interface - delegates to call()"""
        try:
            from rllm.tools.tool_base import ToolOutput
        except ImportError:
            from rllm_mllm.rllm.tools.tool_base import ToolOutput

        try:
            result = await self.call(**kwargs)
            return ToolOutput(name=self.name, output=result)
        except Exception as e:
            return ToolOutput(name=self.name, error=f"{type(e).__name__} - {str(e)}")


class SearchTool(DeepResearchTool):
    """Web search tool using Serper API (ported from Tongyi)."""

    def __init__(self):
        super().__init__(
            name="search",
            description="Performs batched web searches: supply an array 'query'; the tool retrieves the top 10 results for each query in one call.",
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "array",
                        "items": {
                            "type": "string"
                        },
                        "description": "Array of query strings. Include multiple complementary search queries in a single call."
                    },
                },
                "required": ["query"],
            },
        )

    def contains_chinese(self, text: str) -> bool:
        """Check if text contains Chinese characters."""
        return any("\u4e00" <= char <= "\u9fff" for char in text)

    def _zhipu_search(self, query: str | list, api_key: str) -> str:
        """Use Zhipu web_search API when key is available."""
        try:
            import requests
        except ImportError:
            return """[Search - Dependencies Required]

Please install requests: pip install requests"""

        queries = [query] if isinstance(query, str) else query
        all_results: list[str] = []
        proxies = self._get_requests_proxies()

        for q in queries:
            # Check cache for individual query
            cache_key = get_cache_key(q)
            cached_result = get_cache("text_search", cache_key)
            if cached_result:
                all_results.append(cached_result)
                continue
            # Build request
            headers = {
                # Zhipu PaaS expects raw token in Authorization; keep value as-is
                "Authorization": api_key,
                "Content-Type": "application/json",
            }
            location = "us"
            body = {
                "q": q,
                "search_engine": "search_prime",
                "location": location,
                "query_rewrite": False,
                "content_size": "high",
            }

            def send_request():
                return requests.post(
                    "https://search-svip.bigmodel.cn/api/paas/v4/search",
                    headers=headers,
                    data=json.dumps(body, ensure_ascii=False),
                    timeout=300,
                    proxies=proxies,
                )

            try:
                resp = run_with_retries(send_request)
            except Exception as exc:  # noqa: BLE001
                error_message = f"Search request failed for '{q}': {exc}"
                log_search("Zhipu", "Exception", q, error=error_message)
                all_results.append(error_message)
                continue

            text = resp.text
            try:
                data_obj = resp.json()
            except Exception:
                data_obj = None

            if resp.status_code != 200:
                error_message = f"HTTP {resp.status_code}: {text}"
                log_search("Zhipu", "HTTPError", q, error=error_message)
                all_results.append(f"Search returned HTTP {resp.status_code} for '{q}'\n{text}")
                continue

            items = []
            if isinstance(data_obj, dict):
                items = data_obj.get("search_result") or data_obj.get("data") or []

            web_snippets: list[str] = []
            for idx, item in enumerate(items[:10], 1):
                title = item.get("title", "Untitled") if isinstance(item, dict) else "Untitled"
                url = item.get("url", "") if isinstance(item, dict) else ""
                snippet = item.get("description", "") if isinstance(item, dict) else ""
                date = item.get("date") if isinstance(item, dict) else None

                snippet = (snippet or "").strip()

                entry = f"{idx}. [{title}]({url})"
                if date:
                    entry += f"\n   Date published: {date}"
                if snippet:
                    entry += f"\n   {snippet}"
                web_snippets.append(entry)

            content = (
                f"Search for '{q}' returned {len(web_snippets)} results:\n\n" + "\n\n".join(web_snippets)
                if web_snippets
                else f"No search results found for '{q}'"
            )
            log_search("Zhipu", "Success", q, content)
            all_results.append(content)

            # Store individual query result in cache
            set_cache("text_search", cache_key, q, content)

        final_result = "\n=======\n".join(all_results) if len(all_results) > 1 else (all_results[0] if all_results else "")

        return final_result

    def _google_search_fallback(self, query: str | list) -> str:
        """Use Google Custom Search API as fallback."""
        try:
            import requests

            google_key = os.getenv("GOOGLE_SEARCH_SECRET_KEY")
            engine_id = os.getenv("GOOGLE_SEARCH_ENGINE_ID")

            queries = [query] if isinstance(query, str) else query
            all_results = []

            for q in queries:
                params = {"key": google_key, "cx": engine_id, "q": q, "num": 10}

                def send_request():
                    return requests.get(
                        "https://customsearch.googleapis.com/customsearch/v1",
                        params=params,
                        timeout=5,
                    )

                try:
                    response = run_with_retries(send_request)
                except Exception as exc:  # noqa: BLE001
                    error_message = f"Google Custom Search failed for '{q}': {exc}"
                    log_search("Google", "Exception", q, error=error_message)
                    all_results.append(error_message)
                    continue

                if response.status_code == 200:
                    data = response.json()
                    items = data.get("items", [])

                    web_snippets = []
                    for idx, item in enumerate(items[:10], 1):
                        title = item.get("title", "")
                        link = item.get("link", "")
                        snippet = item.get("snippet", "")
                        entry = f"{idx}. [{title}]({link})\n   {snippet}"
                        web_snippets.append(entry)

                    result = f"Google Custom Search for '{q}' returned {len(web_snippets)} results:\n\n" + "\n\n".join(web_snippets)
                    log_search("Google", "Success", q, result)
                    all_results.append(result)
                else:
                    error_message = f"HTTP {response.status_code}: {response.text}"
                    log_search("Google", "HTTPError", q, error=error_message)
                    all_results.append(f"Google Custom Search returned HTTP {response.status_code} for '{q}'")

            return "\n=======\n".join(all_results) if len(all_results) > 1 else all_results[0]

        except Exception as e:
            error_message = f"Google Custom Search error: {e}"
            log_search("Google", "Exception", str(query), error=error_message)
            return error_message

    async def call(self, query: str | list, **kwargs) -> str:
        """
        Search the web using Serper API or Google Custom Search.

        Args:
            query: Search query string or list of queries

        Returns:
            Formatted search results
        """
        # Prefer Zhipu if key available
        zhipu_key = os.getenv("ZHIPU_API_KEY")
        if zhipu_key:
            return self._zhipu_search(query, zhipu_key)

        api_key = os.getenv("SERPER_API_KEY")

        # Try Google Custom Search as fallback if no Serper key
        if not api_key:
            google_key = os.getenv("GOOGLE_SEARCH_SECRET_KEY")
            google_engine_id = os.getenv("GOOGLE_SEARCH_ENGINE_ID")

            if google_key and google_engine_id:
                return self._google_search_fallback(query)

            message = f"""[Search - API Key Required]

To enable real web search, use one of these options:

Option 1 - Serper (Recommended, simpler):
1. Get a free API key from https://serper.dev (2500 searches/month free)
2. Add to .env: SERPER_API_KEY=your_key_here

Option 2 - Google Custom Search:
1. Set up at https://developers.google.com/custom-search
2. Add to .env:
   GOOGLE_SEARCH_SECRET_KEY=your_key
   GOOGLE_SEARCH_ENGINE_ID=your_engine_id

Placeholder results for '{query}'..."""

            log_search("Serper", "Config", str(query), error=message)
            return message

        # Handle single query or list
        queries = [query] if isinstance(query, str) else query
        all_results = []

        for q in queries:
            try:
                if self.contains_chinese(q):
                    payload = json.dumps({"q": q, "location": "China", "gl": "cn", "hl": "zh-cn"})
                else:
                    payload = json.dumps({"q": q, "location": "United States", "gl": "us", "hl": "en"})

                headers = {"X-API-KEY": api_key, "Content-Type": "application/json"}

                def fetch():
                    conn = http.client.HTTPSConnection("google.serper.dev")
                    conn.request("POST", "/search", payload, headers)
                    response = conn.getresponse()
                    status = response.status
                    data = response.read()
                    conn.close()
                    return status, data

                status, data = run_with_retries(fetch)

                if status != 200:
                    error_body = data.decode("utf-8", errors="ignore") if isinstance(data, (bytes, bytearray)) else str(data)
                    error_message = f"HTTP {status}: {error_body}"
                    log_search("Serper", "HTTPError", q, error=error_message)
                    all_results.append(f"Serper search returned HTTP {status} for '{q}'")
                    continue

                results = json.loads(data.decode("utf-8"))

                if "organic" not in results:
                    no_result_message = f"No results found for '{q}'"
                    log_search("Serper", "NoResult", q, result=no_result_message)
                    all_results.append(no_result_message)
                    continue

                web_snippets = []
                for idx, page in enumerate(results.get("organic", [])[:10], 1):
                    date_published = f"\nDate: {page['date']}" if "date" in page else ""
                    source = f"\nSource: {page['source']}" if "source" in page else ""
                    snippet = f"\n{page['snippet']}" if "snippet" in page else ""

                    entry = f"{idx}. [{page.get('title', 'Untitled')}]({page.get('link', '')}){date_published}{source}{snippet}"
                    web_snippets.append(entry)

                content = f"Serper search for '{q}' returned {len(web_snippets)} results:\n\n" + "\n\n".join(web_snippets)
                log_search("Serper", "Success", q, content)
                all_results.append(content)

            except Exception as e:
                error_message = f"Serper search failed for '{q}': {e}"
                log_search("Serper", "Exception", q, error=error_message)
                all_results.append(error_message)

        return "\n=======\n".join(all_results) if len(all_results) > 1 else all_results[0]


class ScholarTool(DeepResearchTool):
    """Google Scholar search using Serper API (ported from Tongyi)."""

    def __init__(self):
        super().__init__(
            name="Scholar",
            description="Search Google Scholar for academic papers",
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The academic search query",
                    }
                },
                "required": ["query"],
            },
        )

    async def call(self, query: str | list, **kwargs) -> str:
        """
        Search Google Scholar using Serper API.

        Args:
            query: Search query string or list of queries

        Returns:
            Academic search results
        """
        api_key = os.getenv("SERPER_API_KEY")
        if not api_key:
            message = """[Scholar - API Key Required]

To enable Google Scholar search, configure SERPER_API_KEY in your .env file."""
            log_search("Scholar", "Config", str(query), error=message)
            return message

        queries = [query] if isinstance(query, str) else query
        all_results = []

        for q in queries:
            try:
                payload = json.dumps({"q": q, "type": "scholar", "num": 10})
                headers = {"X-API-KEY": api_key, "Content-Type": "application/json"}

                def fetch():
                    conn = http.client.HTTPSConnection("google.serper.dev")
                    conn.request("POST", "/scholar", payload, headers)
                    response = conn.getresponse()
                    status = response.status
                    data = response.read()
                    conn.close()
                    return status, data

                status, data = run_with_retries(fetch)

                if status != 200:
                    error_body = data.decode("utf-8", errors="ignore") if isinstance(data, (bytes, bytearray)) else str(data)
                    error_message = f"HTTP {status}: {error_body}"
                    log_search("Scholar", "HTTPError", q, error=error_message)
                    all_results.append(f"Serper Scholar returned HTTP {status} for '{q}'")
                    continue

                results = json.loads(data.decode("utf-8"))

                if "organic" not in results:
                    no_result_message = f"No scholar results found for '{q}'"
                    log_search("Scholar", "NoResult", q, result=no_result_message)
                    all_results.append(no_result_message)
                    continue

                papers = []
                for idx, paper in enumerate(results.get("organic", [])[:10], 1):
                    title = paper.get("title", "Untitled")
                    link = paper.get("link", "")
                    snippet = paper.get("snippet", "")
                    publication = paper.get("publication", "")
                    year = paper.get("year", "")
                    cited_by = paper.get("citedBy", {}).get("value", 0)

                    entry = f"{idx}. [{title}]({link})"
                    if publication:
                        entry += f"\n   Publication: {publication}"
                    if year:
                        entry += f" ({year})"
                    if cited_by:
                        entry += f"\n   Cited by: {cited_by}"
                    if snippet:
                        entry += f"\n   {snippet}"

                    papers.append(entry)

                result_text = f"Serper Scholar search for '{q}':\n\n" + "\n\n".join(papers)
                log_search("Scholar", "Success", q, result_text)
                all_results.append(result_text)

            except Exception as e:
                error_message = f"Serper Scholar request failed for '{q}': {e}"
                log_search("Scholar", "Exception", q, error=error_message)
                all_results.append(error_message)

        return "\n=======\n".join(all_results) if len(all_results) > 1 else all_results[0]


class VisitTool(DeepResearchTool):
    """Web page visiting with content extraction."""

    DEFAULT_READER_URL = "https://search-svip.bigmodel.cn/api/paas/v4/reader"
    DEFAULT_READER_TIMEOUT = 30
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
                        "items": {
                            "type": "string"
                        },
                        "minItems": 1,
                        "description": "The URL(s) of the webpage(s) to visit. Can be a single URL or an array of URLs."
                    },
                    "goal": {
                        "type": "string",
                        "description": "The goal of the visit for webpage(s)."
                        }
                    },
                "required": ["url", "goal"]
            },
        )

    async def call(self, url: str | list, goal: str = "", **kwargs) -> str:
        """Visit webpages via Reader API and optionally summarize with a local vLLM."""

        urls = [url] if isinstance(url, str) else url
        if not urls:
            return "[Visit] No valid URL provided"

        results = []
        for target_url in urls[: self.MAX_URLS]:
            result = await self._handle_single_url(target_url, goal)
            results.append(result)

        return "\n\n=======\n\n".join(results)

    async def _handle_single_url(self, url: str, goal: str) -> str:
        normalized_url = self._normalize_url(url)

        try:
            reader_payload = await asyncio.to_thread(self._fetch_reader_content, normalized_url)
        except Exception as exc:  # noqa: BLE001
            log_tool_event(
                source="Visit/Reader",
                status="Exception",
                message=f"url={normalized_url}",
                error=str(exc),
                level="ERROR",
            )
            return self._build_failure_message(normalized_url, goal, f"Unable to fetch webpage content: {exc}")

        if reader_payload is None:
            return self._build_failure_message(normalized_url, goal, "Reader API returned empty payload")

        content = reader_payload.get("content") or ""
        description = reader_payload.get("description") or ""

        if not content:
            fallback = description or "Webpage content is empty"
            return self._build_failure_message(normalized_url, goal, fallback)

        content = self._truncate_content(content)

        summary_result = await asyncio.to_thread(
            self._summarize_with_vllm,
            content,
            goal,
            reader_payload,
        )

        if summary_result is None:
            evidence_text = content
            summary_text = description or "Summary service unavailable. Returning raw content."
        else:
            evidence_text = summary_result.get("evidence") or content
            summary_text = summary_result.get("summary") or description or ""

        return self._format_success(normalized_url, goal, evidence_text, summary_text)

    def _normalize_url(self, url: str) -> str:
        from urllib.parse import urlparse

        parsed = urlparse(url)
        if not parsed.scheme:
            return f"https://{url}"
        return url

    def _fetch_reader_content(self, url: str) -> dict[str, Any] | None:
        # Check cache first
        cache_key = get_cache_key(url)
        cached_result = get_cache("text_visit", cache_key)
        if cached_result:
            try:
                return json.loads(cached_result)
            except json.JSONDecodeError:
                pass  # Continue with API call if cache is corrupted

        try:
            import requests
        except ImportError as exc:  # noqa: PERF203
            raise RuntimeError("Visit tool requires 'requests' package") from exc

        reader_url = self.DEFAULT_READER_URL
        timeout = int(self.DEFAULT_READER_TIMEOUT)
        authorization = os.getenv("ZHIPU_API_KEY")

        headers = {
            "Content-Type": "application/json",
        }
        if authorization:
            headers["Authorization"] = authorization

        # Support optional headers consistent with the demo scripts
        optional_headers = {
            "X-Return-Format": "markdown",
            "X-No-Cache": "false",
            "X-Timeout": "300",
            "X-Retain-Images": "false",
            "X-With-Images-Summary": "false",
            "X-With-Links-Summary": "false",
        }
        headers.update({k: v for k, v in optional_headers.items() if v is not None})

        body = {
            "url": url,
        }

        proxies = self._get_requests_proxies()

        def send_request():
            return requests.post(
                reader_url,
                headers=headers,
                data=json.dumps(body, ensure_ascii=False),
                timeout=timeout,
                proxies=proxies,
            )

        response = run_with_retries(send_request)

        if response.status_code != 200:
            raise RuntimeError(f"Reader API returned HTTP {response.status_code}")

        try:
            payload = response.json()
        except json.JSONDecodeError as exc:  # noqa: PERF203
            raise RuntimeError("Reader API returned non-JSON payload") from exc

        if not isinstance(payload, dict):
            raise RuntimeError("Reader API payload structure is invalid")

        if payload.get("code") != 200:
            raise RuntimeError(f"Reader API returned error code: {payload.get('code')}")

        data = payload.get("data")
        if not isinstance(data, dict):
            raise RuntimeError("Reader API data field missing or malformed")

        result = {
            "content": data.get("content") or "",
            "description": data.get("description") or "",
            "meta": data,
        }

        # Store result in cache
        set_cache("text_visit", cache_key, url, json.dumps(result, ensure_ascii=False))

        return result

    def _truncate_content(self, content: str) -> str:
        if len(content) <= self.MAX_CONTENT_CHARS:
            return content
        return content[: self.MAX_CONTENT_CHARS] + "\n[Content truncated...]"

    def _summarize_with_vllm(self, content: str, goal: str, reader_payload: dict[str, Any]) -> dict[str, Any] | None:
        vllm_url = os.getenv("VLLM_EXTRACT_URL")
        if not vllm_url:
            log_tool_event(
                source="Visit/vLLM",
                status="Config",
                message="VLLM_EXTRACT_URL is not set, skip summarization service",
            )
            return None
        
        if not re.search(r"/v1/chat/completions/?$", vllm_url):
            vllm_url = f"{vllm_url.rstrip('/')}/v1/chat/completions"

        try:
            import requests
        except ImportError:
            log_tool_event(
                source="Visit/vLLM",
                status="DependencyMissing",
                message="'requests' package not installed, cannot call local vLLM",
                level="WARNING",
            )
            return None

        prompt = self.EXTRACTOR_PROMPT.format(webpage_content=content, goal=goal or "N/A")

        extract_model = os.getenv("EXTRACT_MODEL", "Qwen3-VL-30B-A3B-Instruct")
        max_tokens = int(os.getenv("EXTRACT_MAX_TOKENS", "16384"))
        
        extract_messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]

        if extract_model:
            payload = {
                "model": extract_model,
                "messages": extract_messages,
                "max_tokens": max_tokens,
            }
            
        headers = {"Content-Type": "application/json"}
        proxies = self._get_requests_proxies()

        try:
            response = run_with_retries(
                lambda: requests.post(
                    url=vllm_url,
                    headers=headers,
                    json=payload,
                    timeout=int(os.getenv("VLLM_EXTRACT_TIMEOUT", "300")),
                    proxies=proxies,
                )
            )
        except Exception as exc:  # noqa: BLE001
            log_tool_event(
                source="Visit/vLLM",
                status="RequestError",
                message=f"url={vllm_url}",
                error=str(exc),
                level="ERROR",
            )
            return None

        if response.status_code != 200:
            log_tool_event(
                source="Visit/vLLM",
                status="HTTPError",
                message=f"url={vllm_url} status={response.status_code}",
                level="WARNING",
            )
            return None

        try:
            result = response.json()
        except json.JSONDecodeError:
            log_tool_event(
                source="Visit/vLLM",
                status="ParseError",
                message="vLLM returned non-JSON response, unable to parse",
                level="WARNING",
            )
            return None
        
        log_tool_event(
            source="Visit/vLLM",
            status="Response",
            message=f"raw={json.dumps(result, ensure_ascii=False)}",
        )

        raw_payload: str | dict | None = None
        content_source: str | None = None

        if isinstance(result, dict):
            choices = result.get("choices")
            if isinstance(choices, list) and choices:
                first_choice = choices[0] or {}
                if isinstance(first_choice, dict):
                    message_dict = first_choice.get("message")
                    if isinstance(message_dict, dict):
                        message_content = message_dict.get("content")
                        if isinstance(message_content, str) and message_content.strip():
                            raw_payload = message_content
                            content_source = "choices[0].message.content"
                    if raw_payload is None:
                        text_candidate = first_choice.get("text")
                        if isinstance(text_candidate, str) and text_candidate.strip():
                            raw_payload = text_candidate
                            content_source = "choices[0].text"
            if raw_payload is None:
                fallback_payload = result.get("content") or result.get("data")
                if isinstance(fallback_payload, (str, dict)):
                    raw_payload = fallback_payload
                    content_source = "response.content/data"

        if raw_payload is None:
            log_tool_event(
                source="Visit/vLLM",
                status="InvalidContent",
                message="vLLM response missing usable content",
                level="WARNING",
            )
            return None

        fallback_used = False
        content_dict: dict | None = None

        if isinstance(raw_payload, dict):
            content_dict = raw_payload
        elif isinstance(raw_payload, str):
            candidate = raw_payload.strip()
            if candidate.startswith("`"):
                candidate = candidate.strip("`")
            try:
                content_dict = json.loads(candidate)
                content_source = f"{content_source or 'string_payload'} -> json.loads"
            except json.JSONDecodeError:
                fallback_used = True
                summary_text = candidate
                content_dict = {
                    "rational": "",
                    "evidence": summary_text,
                    "summary": summary_text,
                }
                log_tool_event(
                    source="Visit/vLLM",
                    status="PlainTextFallback",
                    message=(
                        f"content_source={content_source or 'N/A'} "
                        f"summary_len={len(summary_text)} "
                        f"preview={json.dumps(shorten_for_log(summary_text), ensure_ascii=False)}"
                    ),
                    level="INFO",
                )

        if not isinstance(content_dict, dict):
            log_tool_event(
                source="Visit/vLLM",
                status="InvalidContent",
                message="vLLM response does not contain JSON summary content",
                level="WARNING",
            )
            return None

        log_tool_event(
            source="Visit/vLLM",
            status="ParsedSummary",
            message=json.dumps(
                {
                    "model": extract_model or "legacy",
                    "source": content_source,
                    "keys": sorted(content_dict.keys()),
                    "fallback": fallback_used,
                },
                ensure_ascii=False,
            ),
        )

        return content_dict

    def _build_failure_message(self, url: str, goal: str, reason: str) -> str:
        useful_information = f"The useful information in {url} for user goal {goal or 'N/A'} as follows: \n\n"
        useful_information += "Evidence in page: \n" + reason + "\n\n"
        useful_information += "Summary: \n" + "Unable to retrieve webpage content. Please check the link or try again later." + "\n\n"

        reason_preview = shorten_for_log(reason)
        result_preview = shorten_for_log(useful_information)
        log_tool_event(
            source="Visit",
            status="Failure",
            message=(
                f"url={url} "
                f"reason_len={len(reason)} "
                f"result_len={len(useful_information)} "
                f"reason_preview={json.dumps(reason_preview, ensure_ascii=False)} "
                f"result_preview={json.dumps(result_preview, ensure_ascii=False)}"
            ),
            level="WARNING",
        )

        return useful_information

    def _format_success(self, url: str, goal: str, evidence: str, summary: str) -> str:
        useful_information = f"The useful information in {url} for user goal {goal or 'N/A'} as follows: \n\n"
        useful_information += "Evidence in page: \n" + evidence + "\n\n"
        useful_information += "Summary: \n" + (summary or "No summary generated") + "\n\n"

        evidence_text = evidence or ""
        summary_text = summary or ""
        evidence_preview = shorten_for_log(evidence_text)
        summary_preview = shorten_for_log(summary_text)
        log_tool_event(
            source="Visit",
            status="Success",
            message=(
                f"url={url} "
                f"evidence_len={len(evidence_text)} "
                f"summary_len={len(summary_text)} "
                f"evidence_preview={json.dumps(evidence_preview, ensure_ascii=False)} "
                f"summary_preview={json.dumps(summary_preview, ensure_ascii=False)}"
            ),
        )

        return useful_information


class FileParserTool(DeepResearchTool):
    """Enhanced file parsing for multiple formats."""

    def __init__(self):
        super().__init__(
            name="FileParser",
            description="Parse files: TXT, JSON, CSV, PDF, DOCX, etc.",
            parameters={
                "type": "object",
                "properties": {
                    "files": {
                        "type": "string",
                        "description": "File path or list of file paths to parse",
                    }
                },
                "required": ["files"],
            },
        )

    async def call(self, files: str | list, **kwargs) -> str:
        """
        Parse files and extract content.

        Args:
            files: File path string or list of paths

        Returns:
            Extracted file content
        """
        import csv
        from pathlib import Path

        file_paths = [files] if isinstance(files, str) else files
        all_results = []

        for file_path in file_paths[:10]:  # Limit to 10 files
            if not os.path.exists(file_path):
                all_results.append(f"Error: File not found at {file_path}")
                continue

            try:
                file_ext = Path(file_path).suffix.lower()
                file_name = os.path.basename(file_path)
                file_size = os.path.getsize(file_path)

                content = ""

                # Text files
                if file_ext in [
                    ".txt",
                    ".md",
                    ".log",
                    ".py",
                    ".js",
                    ".java",
                    ".cpp",
                    ".c",
                    ".h",
                ]:
                    with open(file_path, encoding="utf-8", errors="ignore") as f:
                        content = f.read()

                # JSON files
                elif file_ext == ".json":
                    with open(file_path, encoding="utf-8") as f:
                        data = json.load(f)
                        content = json.dumps(data, indent=2, ensure_ascii=False)

                # CSV files
                elif file_ext == ".csv":
                    rows = []
                    with open(file_path, encoding="utf-8", errors="ignore") as f:
                        reader = csv.reader(f)
                        for i, row in enumerate(reader):
                            if i >= 100:
                                rows.append("[... truncated ...]")
                                break
                            rows.append(", ".join(row))
                    content = "\n".join(rows)

                # PDF files
                elif file_ext == ".pdf":
                    try:
                        import PyPDF2

                        with open(file_path, "rb") as f:
                            pdf_reader = PyPDF2.PdfReader(f)
                            pages = []
                            for i in range(min(len(pdf_reader.pages), 10)):
                                page = pdf_reader.pages[i]
                                pages.append(f"Page {i + 1}:\n{page.extract_text()}")
                            content = "\n\n".join(pages)
                    except ImportError:
                        content = "[PDF parsing requires: pip install PyPDF2]"

                # Word documents
                elif file_ext in [".docx", ".doc"]:
                    try:
                        from docx import Document

                        doc = Document(file_path)
                        paragraphs = []
                        for i, para in enumerate(doc.paragraphs):
                            if i >= 100:
                                paragraphs.append("[... truncated ...]")
                                break
                            if para.text.strip():
                                paragraphs.append(para.text)
                        content = "\n\n".join(paragraphs)
                    except ImportError:
                        content = "[DOCX parsing requires: pip install python-docx]"

                # Default: try as text
                else:
                    try:
                        with open(file_path, encoding="utf-8", errors="ignore") as f:
                            content = f.read()
                    except Exception:
                        content = f"[Cannot parse file type: {file_ext}]"

                # Limit content
                if len(content) > 10000:
                    content = content[:10000] + "\n[Content truncated...]"

                result = f"[File: {file_name}]\nType: {file_ext}\nSize: {file_size:,} bytes\n\nContent:\n{content}"
                all_results.append(result)

            except Exception as e:
                all_results.append(f"Error parsing {file_path}: {e}")

        return "\n\n=======\n\n".join(all_results)


class PythonInterpreterTool(DeepResearchTool):
    """Safe Python code execution (from existing implementation)."""

    def __init__(self):
        super().__init__(
            name="PythonInterpreter",
            description='Execute Python code in a sandboxed environment. Use this to run Python code and get the execution results.\n**Make sure to use print() for any output you want to see in the results.**\nFor code parameters, use placeholders first, and then put the code within <code></code> XML tags, such as:\n<tool_call>\n{"purpose": <detailed-purpose-of-this-tool-call>, "name": <tool-name>, "arguments": {"code": ""}}\n<code>\nHere is the code.\n</code>\n</tool_call>\n',
            parameters={
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "The Python code to execute. Must be provided within <code></code> XML tags. Remember to use print() statements for any output you want to see.",
                    }
                },
                "required": ["code"],
            },
        )
        self.timeout = 50

    async def call(self, code: str, timeout: int = None, **kwargs) -> str:
        """Execute Python code safely with timeout."""
        timeout = timeout or self.timeout

        code_len = len(code or "")

        def log_result(status: str, message: str, extra: str | None = None, *, level: str | None = None) -> None:
            preview = shorten_for_log(message)
            details = (
                f"code_len={code_len} result_len={len(message)} preview={json.dumps(preview, ensure_ascii=False)}"
            )
            if extra:
                details += f" {extra}"
            log_tool_event(
                source="PythonInterpreter",
                status=status,
                message=details,
                level=level or "INFO",
            )

        # Security checks - check for dangerous imports/operations
        dangerous_patterns = [
            "import os",
            "import subprocess",
            "import sys",
            "from os import",
            "from subprocess import",
            "from sys import",
            "exec(",
            "eval(",
            "compile(",
            "open(",
            "file(",
        ]

        code_lower = code.lower()
        for pattern in dangerous_patterns:
            if pattern in code_lower:
                result = f"[Security Error] '{pattern}' not allowed for safety reasons"
                log_result(
                    "SecurityBlocked",
                    result,
                    extra=f"pattern={json.dumps(pattern, ensure_ascii=False)}",
                    level="WARNING",
                )
                return result

        log_tool_event(
            source="PythonInterpreter",
            status="Security",
            message=f"code_len={code_len} status=passed",
        )

        import io
        import sys
        from concurrent.futures import ThreadPoolExecutor, TimeoutError

        # Setup safe environment
        allowed_modules = {
            "math": __import__("math"),
            "datetime": __import__("datetime"),
            "json": __import__("json"),
            "random": __import__("random"),
            "re": __import__("re"),
            "collections": __import__("collections"),
            "itertools": __import__("itertools"),
            "statistics": __import__("statistics"),
        }

        # Add numpy/pandas if available
        try:
            import numpy as np

            allowed_modules["numpy"] = np
            allowed_modules["np"] = np
        except ImportError:
            pass

        try:
            import pandas as pd

            allowed_modules["pandas"] = pd
            allowed_modules["pd"] = pd
        except ImportError:
            pass

        # Restricted builtins with safe import capability
        def safe_import(name, *args, **kwargs):
            """Allow importing only safe modules."""
            safe_modules = [
                "math",
                "datetime",
                "json",
                "random",
                "re",
                "collections",
                "itertools",
                "statistics",
                "numpy",
                "pandas",
                "scipy",
                "scipy.linalg",  # Add scipy submodules
                "scipy.optimize",
                "scipy.signal",
                "scipy.special",
                "matplotlib",
                "matplotlib.pyplot",
                "urllib.request",
                "requests",
                "sys"
            ]
            # Check if the module or its parent is allowed
            if name in safe_modules or any(name.startswith(m + ".") for m in safe_modules):
                return __import__(name, *args, **kwargs)
            else:
                raise ImportError(f"Module '{name}' is not allowed for safety reasons")

        restricted_builtins = {
            "abs": abs,
            "all": all,
            "any": any,
            "bin": bin,
            "bool": bool,
            "chr": chr,
            "dict": dict,
            "enumerate": enumerate,
            "filter": filter,
            "float": float,
            "hex": hex,
            "int": int,
            "len": len,
            "list": list,
            "map": map,
            "max": max,
            "min": min,
            "oct": oct,
            "ord": ord,
            "pow": pow,
            "print": print,
            "range": range,
            "reversed": reversed,
            "round": round,
            "set": set,
            "slice": slice,
            "sorted": sorted,
            "str": str,
            "sum": sum,
            "tuple": tuple,
            "type": type,
            "zip": zip,
            "__import__": safe_import,  # Allow safe imports
            # Add exception classes for proper error handling
            "Exception": Exception,
            "ImportError": ImportError,
            "ValueError": ValueError,
            "TypeError": TypeError,
            "KeyError": KeyError,
            "IndexError": IndexError,
            "AttributeError": AttributeError,
        }

        global_vars = {"__builtins__": restricted_builtins}
        global_vars.update(allowed_modules)
        local_vars = {}

        # Capture output
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        stdout_buffer = io.StringIO()
        stderr_buffer = io.StringIO()

        def execute_with_timeout():
            try:
                sys.stdout = stdout_buffer
                sys.stderr = stderr_buffer
                exec(code, global_vars, local_vars)
                return True
            except Exception as e:
                stderr_buffer.write(f"Execution error: {e}")
                return False
            finally:
                sys.stdout = old_stdout
                sys.stderr = old_stderr

        # Execute with timeout
        with ThreadPoolExecutor() as executor:
            try:
                future = executor.submit(execute_with_timeout)
                future.result(timeout=timeout)

                stdout_content = stdout_buffer.getvalue()
                stderr_content = stderr_buffer.getvalue()

                if stderr_content:
                    result = f"[Error]\n{stderr_content}"
                    log_result("Error", result, level="ERROR")
                    return result
                elif stdout_content:
                    cleaned_output = stdout_content.rstrip()
                    result = f"[Output]\n{cleaned_output}"
                    log_result("Output", result)
                    return result
                else:
                    meaningful_vars = {k: v for k, v in local_vars.items() if not k.startswith("_") and k not in allowed_modules}
                    if meaningful_vars:
                        result = f"[Variables]\n{meaningful_vars}"
                        log_result("Variables", result)
                        return result
                    else:
                        result = "[Success] Code executed (no output)"
                        log_result("Success", result)
                        return result

            except TimeoutError:
                result = f"[Timeout] Execution exceeded {timeout}s"
                log_result("Timeout", result, level="WARNING")
                return result

        result = "[Error] Unexpected execution error"
        log_result("UnexpectedError", result, level="ERROR")
        return result


class CropAndSearchTool(DeepResearchTool):
    """Crop and search tool for visual deep research."""

    # Concurrency limits to prevent API rate limiting
    MAX_CONCURRENT_BBOX = 5  # Limit concurrent bbox processing
    MAX_CONCURRENT_WEBPAGE_VISITS = 10  # Limit concurrent webpage visits per bbox

    def __init__(self):
        # Note: reference version raised ImportError here when PIL/requests/oss2
        # are unavailable. We defer that check to `.call()` so that the rest of
        # the toolset (search/visit/python) can still be loaded in lightweight
        # environments where crop_and_search is not exercised.
        super().__init__(
            name="crop_and_search",
            description="Crop regions from an image and perform visual search to gather information. Takes an image_id (path or URL), bbox coordinates (single or multiple), and goal description.",
            parameters={
                "type": "object",
                "properties": {
                    "image_id": {
                        "type": "string",
                        "description": "Path or URL of the image to process"
                    },
                    "bbox": {
                        "type": "array",
                        "items": {
                            "anyOf": [
                                {"type": "array", "items": {"type": "number"}, "minItems": 4, "maxItems": 4},
                                {"type": "number"}
                            ]
                        },
                        "description": "Bounding box coordinates [x1,y1,x2,y2] or array of bboxes"
                    },
                    "goal": {
                        "type": "string",
                        "description": "Description of what to search for in the cropped regions"
                    }
                },
                "required": ["image_id", "bbox"]
            },
        )

    def _crop_image_by_bbox(self, image_path: str, bbox: List[int], output_dir: str) -> Optional[str]:
        """Crop image by bounding box coordinates."""
        try:
            os.makedirs(output_dir, exist_ok=True)

            with Image.open(image_path) as img:
                if img.mode != 'RGB':
                    img = img.convert('RGB')

                width, height = img.size

                # Convert coordinates (assuming bbox is in 0-1000 range)
                x1 = max(0, min(int(bbox[0] * width / 1000), width - 1))
                y1 = max(0, min(int(bbox[1] * height / 1000), height - 1))
                x2 = max(0, min(int(bbox[2] * width / 1000), width - 1))
                y2 = max(0, min(int(bbox[3] * height / 1000), height - 1))

                if x2 <= x1 or y2 <= y1:
                    log_tool_event("CropAndSearch", "InvalidBbox", f"bbox={bbox}", level="WARNING")
                    return None

                # Crop and resize
                cropped_img = img.crop((x1, y1, x2, y2))
                cropped_img = cropped_img.resize(
                    (cropped_img.width * 2, cropped_img.height * 2),
                    Image.Resampling.LANCZOS
                )

                # Generate deterministic filename based on image path and bbox
                # This ensures same image_id + bbox always produces same filename for caching
                image_basename = os.path.basename(image_path)
                image_name_no_ext = os.path.splitext(image_basename)[0]
                bbox_str = f"{bbox[0]}_{bbox[1]}_{bbox[2]}_{bbox[3]}"
                timestamp = int(time.time())
                base_name = f"crop_{timestamp}"
                deterministic_name = f"{base_name}_{uuid.uuid4().hex[:8]}.jpg"
                # deterministic_name = f"crop_{image_name_no_ext}_{bbox_str}.jpg"

                output_path = os.path.join(output_dir, deterministic_name)
                cropped_img.save(output_path, "JPEG", quality=95)

                return output_path

        except Exception as e:
            log_tool_event("CropAndSearch", "CropError", str(e), level="ERROR")
            return None

    def _upload_to_oss(self, local_path: str) -> Optional[str]:
        """Upload local image to OSS."""
        try:
            filename = os.path.basename(local_path)
            oss_path = filename

            bucket = get_oss_bucket()
            with open(local_path, "rb") as f:
                bucket.put_object(oss_path, f)

            public_url = f"https://{OSS_CONFIG['bucket_name']}.{OSS_CONFIG['endpoint'].replace('https://', '')}/{oss_path}"
            return public_url

        except Exception as e:
            log_tool_event("CropAndSearch", "UploadError", str(e), level="ERROR")
            return None

    def _image_search(self, oss_url: str, max_retries: int = 3) -> Optional[List[Dict[str, str]]]:
        """Perform image search using Zhipu API."""
        # Check cache first
        cache_key = get_cache_key(oss_url)
        cached_result = get_cache("image_search", cache_key)
        if cached_result:
            try:
                return json.loads(cached_result)
            except json.JSONDecodeError:
                pass  # Continue with API call if cache is corrupted

        headers = {
            "Authorization": SEARCH_CONFIG["zhipu_key"],
            "Content-Type": "application/json",
            "Accept": "*/*"
        }

        payload = {"url": oss_url}
        proxies = self._get_requests_proxies()

        def make_search_request():
            response = requests.post(
                SEARCH_CONFIG["search_url"],
                headers=headers,
                json=payload,
                timeout=30,
                proxies=proxies,
            )
            response.raise_for_status()
            return response

        try:
            response = run_with_retries(
                func=make_search_request,
                attempts=max_retries,
                delay=1.0
            )

            result_data = response.json()
            search_results = result_data.get('search_result', [])

            formatted_results = []
            for item in search_results[:3]:  # Take top 3
                title = item.get("title", "Untitled")
                image_url = item.get("image_url", "")
                link = item.get("link", "")
                source = item.get("source", "")
                thumbnail_url = item.get("thumbnail_url", "")

                if image_url and link:
                    formatted_results.append({
                        "title": title,
                        "image_url": image_url,
                        "link": link,
                        "bbox_image_url": oss_url,
                        "source": source,
                        "thumbnail_url": thumbnail_url
                    })

            final_result = formatted_results if formatted_results else None

            # Store result in cache
            if final_result is not None:
                set_cache("image_search", cache_key, oss_url, json.dumps(final_result, ensure_ascii=False))

            return final_result

        except Exception as e:
            log_tool_event("CropAndSearch", "SearchError", f"url={oss_url} error={str(e)}", level="ERROR")
            return None

    # ============================================================================
    # Webpage visiting functions (moved from visit_summary_vl.py)
    # ============================================================================

    def _validate_base64(self, base64_string: str) -> bool:
        """Validate if a base64 string is valid."""
        try:
            # Check if it contains data URI prefix
            if base64_string.startswith('data:image/'):
                # Extract base64 part
                if ';base64,' in base64_string:
                    base64_part = base64_string.split(';base64,', 1)[1]
                else:
                    return False
            else:
                base64_part = base64_string

            # Try to decode
            base64.b64decode(base64_part, validate=True)
            return True
        except Exception:
            return False

    def _encode_local_file_to_base64(self, file_path: str) -> Optional[str]:
        """Encode a local image file to base64 format."""
        try:
            if not os.path.exists(file_path):
                return None

            with open(file_path, 'rb') as image_file:
                extension = file_path.split('.')[-1].lower()
                if extension in ['jpg', 'jpeg']:
                    image_format = 'jpeg'
                elif extension == 'png':
                    image_format = 'png'
                elif extension == 'gif':
                    image_format = 'gif'
                elif extension == 'webp':
                    image_format = 'webp'
                elif extension == 'bmp':
                    image_format = 'bmp'
                else:
                    image_format = 'jpeg'

                encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
                result = f"data:image/{image_format};base64,{encoded_string}"

                if self._validate_base64(result):
                    return result
                else:
                    return None
        except Exception:
            return None

    def _encode_url_to_base64(self, url: str, timeout: int = 30) -> Optional[str]:
        """Encode a network image URL to base64 format."""
        try:
            # Get proxy settings
            proxies = self._get_requests_proxies()

            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'Accept': 'image/webp,image/apng,image/*,*/*;q=0.8',
            }

            response = requests.get(url, timeout=timeout, proxies=proxies, headers=headers, stream=True)
            response.raise_for_status()

            content = b""
            max_size = 10 * 1024 * 1024  # 10 MB limit
            for chunk in response.iter_content(chunk_size=8192):
                content += chunk
                if len(content) > max_size:
                    return None

            if len(content) == 0:
                return None

            # Determine image format
            content_type = response.headers.get('content-type', '')
            image_format = 'jpeg'

            if content_type.startswith('image/'):
                image_format = content_type.split('/')[-1].split(';')[0].lower()
            else:
                # Try to infer from content
                if content[:3] == b'\xff\xd8\xff':
                    image_format = 'jpeg'
                elif content[:8] == b'\x89PNG\r\n\x1a\n':
                    image_format = 'png'
                elif content[:6] in [b'GIF87a', b'GIF89a']:
                    image_format = 'gif'
                elif content[:4] == b'RIFF' and content[8:12] == b'WEBP':
                    image_format = 'webp'

            if image_format not in ['jpeg', 'png', 'gif', 'webp', 'bmp']:
                image_format = 'jpeg'

            encoded_string = base64.b64encode(content).decode('utf-8')
            result = f"data:image/{image_format};base64,{encoded_string}"

            if self._validate_base64(result):
                return result
            else:
                return None

        except Exception:
            return None

    def _safe_encode_image_to_base64(self, image_path: str, timeout: int = 5) -> Optional[str]:
        """Safely encode an image to base64 with validation."""
        try:
            if image_path.startswith(('http://', 'https://')):
                result = self._encode_url_to_base64(image_path, timeout)
            else:
                result = self._encode_local_file_to_base64(image_path)

            if result and self._validate_base64(result):
                return result
            else:
                return None
        except Exception:
            return None

    def _extract_images_from_content(self, content: str) -> List[Tuple[str, str]]:
        """Extract all image alt texts and URLs from webpage content."""
        pattern = r'!\[(.*?)\]\((https?://[^\s]+)\)'
        matches = re.findall(pattern, content)
        return matches

    def _summarize_with_vllm_only_text(self, content: str, goal: str, reader_payload: Dict[str, Any], max_retries: int = 3) -> Optional[Dict[str, Any]]:
        """Text-only version of webpage content summarization."""
        TEXT_ONLY_PROMPT = """You are a text analysis assistant. You will receive webpage content (text only) and a user's goal. Your task is to extract information that helps achieve the user's goal.

## Task Guidelines
1. **Content Relevance**: Evaluate how the webpage text relates to the user's goal.
2. **Information Extraction**: Extract key information from the webpage text that supports the user's goal.

## Final Output Requirements
- Output **only** a valid JSON object (no Markdown, code blocks, or any other text).
- The JSON object must contain three keys: `"rational"`, `"evidence"`, and `"summary"`.
- Each key must map to a **string** (use an empty string if no relevant content is available).
- Do not include any additional fields or explanations outside the JSON object.

Example:
{"rational": "Explain why the information is relevant to the goal.", "evidence": "Quote or paraphrase the key supporting content from the webpage.", "summary": "Provide a concise summary that connects the evidence back to the goal."}
"""

        if not content or not content.strip():
            return {
                "rational": "No valid text content extracted from webpage",
                "evidence": "",
                "summary": "Unable to process webpage content, text content is empty"
            }

        max_text_length = 50000
        truncated_content = content[:max_text_length] + "...\n[Content truncated]" if len(content) > max_text_length else content

        message_content = [
            {"type": "text", "text": f"Webpage content:\n\n{truncated_content}"}
        ]

        if goal:
            message_content.append({"type": "text", "text": f"\nUser's goal: {goal}"})

        messages = [
            {"role": "system", "content": TEXT_ONLY_PROMPT},
            {"role": "user", "content": message_content}
        ]

        # Try to use VLLM if available
        vllm_url = os.getenv("VLLM_EXTRACT_URL")
        if vllm_url:
            try:
                if not re.search(r"/v1/chat/completions/?$", vllm_url):
                    vllm_url = f"{vllm_url.rstrip('/')}/v1/chat/completions"

                extract_model = os.getenv("EXTRACT_MODEL", "Qwen3-VL-30B-A3B-Instruct")
                payload = {
                    "model": extract_model,
                    "messages": messages,
                    "max_tokens": 16384,
                }

                headers = {"Content-Type": "application/json"}
                proxies = self._get_requests_proxies()

                response = run_with_retries(
                    lambda: requests.post(vllm_url, headers=headers, json=payload, timeout=300, proxies=proxies)
                )

                if response.status_code == 200:
                    result = response.json()
                    if isinstance(result, dict):
                        choices = result.get("choices", [])
                        if choices:
                            content = choices[0].get("message", {}).get("content", "")
                            if content:
                                try:
                                    parsed = json.loads(content.strip())
                                    return parsed
                                except:
                                    pass
            except Exception:
                pass

        return None

    def _summarize_with_vllm(self, content: str, goal: str, reader_payload: Dict[str, Any],
                           query_image_url: Optional[str] = None, title: str = "", image_url: str = "",
                           thumbnail_url: str = "", source: str = "", max_images: int = 10) -> Optional[Dict[str, Any]]:
        """Summarize webpage content using visual language model."""
        EXTRACTOR_PROMPT = """You are a multimodal intelligent assistant capable of analyzing both images and text. You will receive a user's query image, query goal, and relevant web content (including search result previews such as website source, page title, and preview images; as well as the main body text and images retrieved from accessing the web pages). Your task is to extract key information that helps the user achieve their goal.

## Task Guidelines
1. **Image Matching**: Compare the user's query image with the images on the web pages (including search result preview images and images within the page content). Evaluate their visual relevance and determine whether they depict the same entity as the user's query image.
2. **Information Extraction**: When they are determined to be the same entity: Extract from the web content the key information most relevant to the user's goal, to support or fulfill the user's query. When they are determined not to be the same entity: Briefly describe the main visual differences between the query image and the web images, and extract information from the web pages that may still be useful as a reference for the user.

## Final Output Requirements
- Output **only** a valid JSON object (no Markdown, code blocks, comments, or any other text).
- The JSON object must contain three keys: `"rational"`, `"evidence"`, and `"summary"`.
- Each key must map to a **string** (use an empty string if no relevant content is available).
- Do not include any other fields or explanations outside the JSON object.

Example:
{{"rational": "Explain why the information is relevant to the goal.", "evidence": "Quote or paraphrase the key supporting content from the webpage.", "summary": "Provide a concise summary that connects the evidence back to the goal."}}
"""

        message_content: List[Dict[str, Any]] = []

        # 1. User's query image
        if query_image_url:
            query_image_base64 = self._safe_encode_image_to_base64(query_image_url)
            if query_image_base64:
                message_content.append({"type": "text", "text": "User's query image (the image the user is searching for):"})
                message_content.append({"type": "image_url", "image_url": {"url": query_image_base64}})

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
            message_content.append({"type": "text", "text": "Search result metadata:\n" + "\n".join(preview_parts)})

        # Preview images
        for label, url in [("Main image", image_url), ("Thumbnail", thumbnail_url)]:
            if url:
                img_b64 = self._safe_encode_image_to_base64(url)
                if img_b64:
                    message_content.append({"type": "text", "text": f"Search result {label}:"})
                    message_content.append({"type": "image_url", "image_url": {"url": img_b64}})

        # 4. Webpage content
        if content.strip():
            max_text_length = 50000
            truncated_content = content[:max_text_length] + "...\n[Content truncated]" if len(content) > max_text_length else content
            message_content.append({"type": "text", "text": "Webpage content:\n\n" + truncated_content})

        # 5. Images from webpage content
        if content:
            image_matches = self._extract_images_from_content(content)
            if image_matches:
                webpage_images = []
                for alt_text, img_url in image_matches[:max_images]:
                    img_base64 = self._safe_encode_image_to_base64(img_url)
                    if img_base64:
                        webpage_images.append((alt_text, img_base64))

                if webpage_images:
                    message_content.append({"type": "text", "text": "Images from webpage:"})
                    for alt_text, img_base64 in webpage_images:
                        if alt_text.strip():
                            message_content.append({"type": "text", "text": f"Image '{alt_text}':"})
                        message_content.append({"type": "image_url", "image_url": {"url": img_base64}})

        # Check if we have content
        has_content = any(
            (item["type"] == "text" and item["text"].strip()) or item["type"] == "image_url"
            for item in message_content
        )

        if not has_content:
            return {
                "rational": "No valid content extracted from webpage or search results.",
                "evidence": "",
                "summary": "Unable to process webpage and search preview content."
            }

        # Try VLLM first
        vllm_url = os.getenv("VLLM_EXTRACT_URL")
        if vllm_url:
            try:
                if not re.search(r"/v1/chat/completions/?$", vllm_url):
                    vllm_url = f"{vllm_url.rstrip('/')}/v1/chat/completions"

                extract_model = os.getenv("EXTRACT_MODEL", "Qwen3-VL-30B-A3B-Instruct")
                messages = [
                    {"role": "system", "content": EXTRACTOR_PROMPT},
                    {"role": "user", "content": message_content}
                ]

                payload = {"model": extract_model, "messages": messages, "max_tokens": 16384}
                headers = {"Content-Type": "application/json"}
                proxies = self._get_requests_proxies()

                response = run_with_retries(
                    lambda: requests.post(vllm_url, headers=headers, json=payload, timeout=300, proxies=proxies)
                )

                if response.status_code == 200:
                    result = response.json()
                    if isinstance(result, dict):
                        choices = result.get("choices", [])
                        if choices:
                            content = choices[0].get("message", {}).get("content", "")
                            if content:
                                try:
                                    cleaned = content.strip()
                                    if cleaned.startswith("```json"):
                                        cleaned = cleaned[7:]
                                    elif cleaned.startswith("```"):
                                        cleaned = cleaned[3:]
                                    if cleaned.endswith("```"):
                                        cleaned = cleaned[:-3]

                                    parsed = json.loads(cleaned)
                                    if isinstance(parsed, dict):
                                        for key in ["rational", "evidence", "summary"]:
                                            if key not in parsed:
                                                parsed[key] = ""
                                        return parsed
                                except:
                                    pass
            except Exception:
                pass

        # Fallback to text-only
        return self._summarize_with_vllm_only_text(content, goal, reader_payload)

    def _fetch_reader_content(self, url: str) -> Optional[Dict[str, Any]]:
        """Fetch webpage content using Reader API."""
        # Check cache first
        cache_key = get_cache_key(url)
        cached_result = get_cache("image_visit", cache_key)
        if cached_result:
            try:
                return json.loads(cached_result)
            except json.JSONDecodeError:
                pass  # Continue with API call if cache is corrupted

        try:
            reader_url = "https://search-svip.bigmodel.cn/api/paas/v4/reader"
            authorization = os.getenv("ZHIPU_API_KEY")

            headers = {"Content-Type": "application/json"}
            if authorization:
                headers["Authorization"] = authorization

            optional_headers = {
                "X-Return-Format": "markdown",
                "X-No-Cache": "false",
                "X-Timeout": "300",
                "X-Retain-Images": "true",
                "X-With-Images-Summary": "true",
                "X-With-Links-Summary": "true",
            }
            headers.update({k: v for k, v in optional_headers.items() if v is not None})

            body = {"url": url}
            proxies = self._get_requests_proxies()

            def send_request():
                return requests.post(reader_url, headers=headers, json=body, timeout=30, proxies=proxies)

            response = run_with_retries(send_request)

            if response.status_code != 200:
                return None

            payload = response.json()
            if payload.get("code") != 200:
                return None

            data = payload.get("data", {})
            result = {
                "content": data.get("content") or "",
                "description": data.get("description") or "",
                "meta": data,
            }

            # Store result in cache
            set_cache("image_visit", cache_key, url, json.dumps(result, ensure_ascii=False))

            return result

        except Exception:
            return None

    async def _handle_single_url(self, url: str, goal: str, query_image_url: Optional[str] = None,
                          title: str = "", thumbnail_url: str = "", image_url: str = "",
                          source: str = "", max_content_chars: int = 120000) -> str:
        """Handle visiting a single URL."""
        try:
            reader_payload = await asyncio.to_thread(self._fetch_reader_content, url)
            if not reader_payload:
                return f"[Error] Failed to fetch content from {url}"

            content = reader_payload.get("content") or ""
            description = reader_payload.get("description") or ""

            if not content:
                content = "Webpage content is empty."

            # Truncate content
            if len(content) > max_content_chars:
                content = content[:max_content_chars] + "\n[Content truncated...]"

            # Try visual summarization
            summary_result = await asyncio.to_thread(
                self._summarize_with_vllm,
                content=content, goal=goal, reader_payload=reader_payload,
                query_image_url=query_image_url, title=title, image_url=image_url,
                thumbnail_url=thumbnail_url, source=source
            )

            if summary_result:
                rational_text = summary_result.get("rational") or ""
                evidence_text = summary_result.get("evidence") or content[:2000] + ("..." if len(content) > 2000 else "")
                summary_text = summary_result.get("summary") or description or ""
            else:
                rational_text = ""
                evidence_text = content[:2000] + ("..." if len(content) > 2000 else "")
                summary_text = description or "Summary unavailable."

            result = f"The useful information in [{title}]({url}) are:\n\n"
            result += f"Evidence in page:\n{evidence_text}\n\n"
            result += f"Summary:\n{summary_text}\n\n"

            return result

        except Exception as e:
            return f"[Error] Failed to process {url}: {str(e)}"

    async def _visit_webpages_for_search(self, search_results: List[Dict[str, str]], goal: str) -> str:
        """Visit webpages for search results and extract relevant information."""
        try:
            # Use semaphore to limit concurrent webpage visits
            semaphore = asyncio.Semaphore(self.MAX_CONCURRENT_WEBPAGE_VISITS)

            async def visit_single_page_with_limit(item):
                async with semaphore:
                    return await self._handle_single_url(
                        url=item['link'],
                        goal=goal,
                        query_image_url=item['bbox_image_url'],
                        title=item['title'],
                        thumbnail_url=item['thumbnail_url'],
                        image_url=item['image_url'],
                        source=item['source']
                    )

            # Create concurrent tasks for all webpage visits with concurrency control
            visit_tasks = [visit_single_page_with_limit(item) for item in search_results]

            # Execute all webpage visits concurrently with controlled parallelism
            log_tool_event("CropAndSearch", "ConcurrentWebpageVisits",
                         f"Visiting {len(visit_tasks)} webpages with max {self.MAX_CONCURRENT_WEBPAGE_VISITS} concurrent")
            visit_results = await asyncio.gather(*visit_tasks, return_exceptions=True)

            # Process results
            all_results = []
            for i, result in enumerate(visit_results):
                try:
                    if isinstance(result, Exception):
                        log_tool_event("CropAndSearch", "VisitTaskException", f"webpage_{i+1} error={str(result)}", level="ERROR")
                        all_results.append(f"[{i+1}] [Error visiting webpage: {str(result)}]")
                    elif isinstance(result, str):
                        # _handle_single_url returns a string directly
                        all_results.append(f"[{i+1}] {result}")
                    else:
                        log_tool_event("CropAndSearch", "InvalidVisitResult", f"webpage_{i+1} unexpected_result_type={type(result)}", level="ERROR")
                        all_results.append(f"[{i+1}] [Invalid result format: {type(result)}]")
                except Exception as e:
                    log_tool_event("CropAndSearch", "VisitResultProcessingError", f"webpage_{i+1} error={str(e)}", level="ERROR")
                    all_results.append(f"[{i+1}] [Error processing visit result: {str(e)}]")

            return "\n\n=======\n\n".join(all_results)

        except Exception as e:
            log_tool_event("CropAndSearch", "VisitSetupError", str(e), level="ERROR")
            return f"[Error setting up webpage visits: {str(e)}]"

    async def _process_single_bbox(self, bbox: List[int], bbox_index: int, image_id: str, cache_dir: str, goal: str) -> Tuple[int, str, Optional[str]]:
        """Process a single bounding box concurrently."""
        log_tool_event("CropAndSearch", "Processing", f"bbox_{bbox_index+1}={bbox}")

        try:
            # 1. Crop image (CPU-intensive, run in thread pool)
            cropped_path = await asyncio.to_thread(self._crop_image_by_bbox, image_id, bbox, cache_dir)
            if not cropped_path:
                return bbox_index, f"Bbox {bbox}: Image cropping failed", None

            # 2. Upload to OSS (I/O bound, run in thread pool)
            oss_url = await asyncio.to_thread(self._upload_to_oss, cropped_path)
            if not oss_url:
                return bbox_index, f"Bbox {bbox}: OSS upload failed", None

            # 3. Perform image search (network I/O, run in thread pool)
            search_results = await asyncio.to_thread(self._image_search, oss_url)
            if not search_results:
                return bbox_index, f"Bbox {bbox}: Image search failed", oss_url

            # 4. Visit webpages and extract information (fully concurrent)
            visit_results = await self._visit_webpages_for_search(search_results, goal)

            result_text = f"The search results for bbox {bbox} are as follows:\n{visit_results}"
            return bbox_index, result_text, oss_url

        except Exception as e:
            log_tool_event("CropAndSearch", "BboxError", f"bbox_{bbox_index+1} error={str(e)}", level="ERROR")
            return bbox_index, f"Bbox {bbox}: Processing failed - {str(e)}", None

    async def call(self, image_id: str, bbox: Union[List[int], List[List[int]]], goal: str = "", **kwargs) -> str:
        """Execute crop and search operation with controlled concurrency."""
        if not PIL_AVAILABLE:
            return "[CropAndSearch] requires PIL, requests, and oss2 packages — install them to enable crop_and_search"
        # Create temporary directory for processing
        cache_dir = os.getenv("IMAGE_CROP_CACHE", None)
        if cache_dir is None:
            raise ValueError("IMAGE_CROP_CACHE must be provided.")
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

            # Use semaphore to limit concurrent bbox processing
            semaphore = asyncio.Semaphore(self.MAX_CONCURRENT_BBOX)

            async def process_bbox_with_limit(single_bbox, i):
                async with semaphore:
                    return await self._process_single_bbox(single_bbox, i, image_id, cache_dir, goal)

            # Create concurrent tasks for all bboxes with concurrency control
            tasks = [
                process_bbox_with_limit(single_bbox, i)
                for i, single_bbox in enumerate(bbox_list)
            ]

            # Execute all tasks concurrently with controlled parallelism
            log_tool_event("CropAndSearch", "ConcurrentProcessing",
                         f"Processing {len(tasks)} bboxes with max {self.MAX_CONCURRENT_BBOX} concurrent")
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Process results and maintain order
            all_results = []
            oss_urls = []

            # Process results while maintaining order
            # Since we created tasks in order, results should be in the same order
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    log_tool_event("CropAndSearch", "TaskException", f"bbox_{i+1} error={str(result)}", level="ERROR")
                    all_results.append(f"Bbox {bbox_list[i]}: Task failed - {str(result)}")
                elif isinstance(result, tuple) and len(result) == 3:
                    bbox_index, result_text, oss_url = result
                    all_results.append(result_text)
                    if oss_url:
                        oss_urls.append(oss_url)
                else:
                    log_tool_event("CropAndSearch", "InvalidResult", f"bbox_{i+1} unexpected_result_type={type(result)}", level="ERROR")
                    all_results.append(f"Bbox {bbox_list[i]}: Invalid result format")

            log_tool_event("CropAndSearch", "Completed", f"Processed {len(bbox_list)} bboxes, {len(oss_urls)} successful uploads")

            return "\n\n=======\n\n".join(all_results)

        except Exception as e:
            log_tool_event("CropAndSearch", "ExecutionError", str(e), level="ERROR")
            return f"[CropAndSearch Error] {str(e)}"


def check_cache_health() -> bool:
    """Check if cache database is healthy and repair if needed."""
    try:
        db = get_cache_db()

        # Test basic connectivity
        cursor = db.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table'")
        table_count = cursor.fetchone()[0]

        # Check if our tables exist
        expected_tables = {"text_search", "text_visit", "image_search", "image_visit"}
        cursor = db.execute("SELECT name FROM sqlite_master WHERE type='table'")
        existing_tables = {row[0] for row in cursor.fetchall()}

        if not expected_tables.issubset(existing_tables):
            missing_tables = expected_tables - existing_tables
            log_tool_event("Cache", "Repair", f"Recreating missing tables: {missing_tables}")
            # Recreate tables
            _create_cache_tables()
            return True

        # Test WAL file size (rough check)
        try:
            wal_path = CACHE_CONFIG["db_path"] + "-wal"
            if os.path.exists(wal_path):
                wal_size = os.path.getsize(wal_path)
                if wal_size > 100 * 1024 * 1024:  # 100MB
                    log_tool_event("Cache", "WALSize", f"WAL file too large: {wal_size} bytes", level="WARNING")
        except:
            pass

        return True

    except Exception as e:
        log_tool_event("Cache", "HealthCheck", f"Health check failed: {str(e)}", level="ERROR")
        return False


def cleanup_expired_cache():
    """Clean up expired cache entries and optimize database."""
    try:
        db = get_cache_db()
        max_age_seconds = CACHE_CONFIG["max_age_days"] * 24 * 60 * 60
        cutoff_time = time.time() - max_age_seconds

        with db:
            # Clean up expired entries
            tables = ["text_search", "text_visit", "image_search", "image_visit"]
            total_deleted = 0

            for table in tables:
                cursor = db.execute(f"DELETE FROM {table} WHERE last_accessed < ?", (cutoff_time,))
                deleted_count = cursor.rowcount
                total_deleted += deleted_count

            if total_deleted > 0:
                log_tool_event("Cache", "Cleanup", f"Deleted {total_deleted} expired entries")

            # Optimize database
            db.execute("VACUUM")  # Reclaim space
            db.execute("ANALYZE")  # Update statistics

    except Exception as e:
        log_tool_event("Cache", "CleanupError", f"Failed to cleanup cache: {str(e)}", level="ERROR")


# Initialize cache database on module import
try:
    get_cache_db()
    _create_cache_tables()  # Ensure tables exist
    check_cache_health()
    # Clean up expired cache entries on startup
    cleanup_expired_cache()
except Exception as e:
    log_tool_event("Cache", "InitError", f"Failed to initialize cache: {str(e)}", level="WARNING")


# Tool registry
DEEPRESEARCH_TOOLS = {
    "search": SearchTool(),
    # "Scholar": ScholarTool(),
    "visit": VisitTool(),
    # "FileParser": FileParserTool(),
    "PythonInterpreter": PythonInterpreterTool(),
    "crop_and_search": CropAndSearchTool(),  # Enable if PIL, requests, oss2 are available
}


def get_tool(name: str) -> DeepResearchTool:
    """Get a tool by name."""
    return DEEPRESEARCH_TOOLS.get(name)


def get_all_tools() -> dict[str, DeepResearchTool]:
    """Get all available tools."""
    return DEEPRESEARCH_TOOLS.copy()
