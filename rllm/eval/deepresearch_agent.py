from __future__ import annotations

import asyncio
import time
from datetime import datetime
from typing import Any, List, Optional

from PIL import Image
import re
from collections import Counter
import json5

from openai import AsyncOpenAI

# Constants from original DeepResearch
OBS_START = "<tool_response>"
OBS_END = "\n</tool_response>"
MAX_LLM_CALL_PER_RUN = 50
MAX_REPEAT_TURN=3

DEEPRESEARCH_SYSTEM_PROMPT_TEXT = """You are a deep research assistant. Your core function is to conduct thorough, multi-source investigations into any topic. You must handle both broad, open-domain inquiries and queries within specialized academic fields. For every request, synthesize information from credible, diverse sources to deliver a comprehensive, accurate, and objective response. When you have gathered sufficient information and are ready to provide the definitive response, you must enclose the entire final answer within <answer></answer> tags.

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"type": "function", "function": {"name": "search", "description": "Perform Google web searches then returns a string of the top search results. Accepts multiple queries.", "parameters": {"type": "object", "properties": {"query": {"type": "array", "items": {"type": "string", "description": "The search query."}, "minItems": 1, "description": "The list of search queries."}}, "required": ["query"]}}}
{"type": "function", "function": {"name": "visit", "description": "Visit webpage(s) and return the summary of the content.", "parameters": {"type": "object", "properties": {"url": {"type": "array", "items": {"type": "string"}, "description": "The URL(s) of the webpage(s) to visit. Can be a single URL or an array of URLs."}, "goal": {"type": "string", "description": "The specific information goal for visiting webpage(s)."}}, "required": ["url", "goal"]}}}
{"type": "function", "function": {"name": "PythonInterpreter", "description": "Executes Python code in a sandboxed environment. To use this tool, you must follow this format:
1. The 'arguments' JSON object must be empty: {}.
2. The Python code to be executed must be placed immediately after the JSON block, enclosed within <code> and </code> tags.

IMPORTANT: Any output you want to see MUST be printed to standard output using the print() function.

Example of a correct call:
<tool_call>
{"name": "PythonInterpreter", "arguments": {}}
<code>
import numpy as np
# Your code here
print(f"The result is: {np.mean([1,2,3])}")
</code>
</tool_call>", "parameters": {"type": "object", "properties": {}, "required": []}}}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>

Current date: """


DEEPRESEARCH_SYSTEM_PROMPT = """You are a deep research assistant. Your core function is to conduct thorough, multi-source investigations into any topic. You must handle both broad, open-domain inquiries and queries within specialized academic fields. For every request, synthesize information from credible, diverse sources to deliver a comprehensive, accurate, and objective response. When you have gathered sufficient information and are ready to provide the definitive response, you must enclose the entire final answer within <answer></answer> tags.

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"type": "function", "function": {"name": "search", "description": "Perform Google web searches then returns a string of the top search results. Accepts multiple queries.", "parameters": {"type": "object", "properties": {"query": {"type": "array", "items": {"type": "string", "description": "The search query."}, "minItems": 1, "description": "The list of search queries."}}, "required": ["query"]}}}
{"type": "function", "function": {"name": "crop_and_search", "description": "Crop some important local regions from an image and perform reverse image / visual search to identify objects, text, organizations, or other visual elements.", "parameters": {"type": "object", "properties": {"image_id": {"type": "string", "description": "The path or unique identifier of the image to analyze."}, "bbox": {"type": "array", "items": {"type": "array", "items": {"type": "number"}, "description": "Bounding box coordinates [x1, y1, x2, y2]."}, "minItems": 1, "description": "One or more important local regions to be cropped from the image."}, "goal": {"type": "string", "description": "The specific purpose of the visual search."}}, "required": ["image_id", "bbox", "goal"]}}}
{"type": "function", "function": {"name": "visit", "description": "Visit webpage(s) and return the summary of the content.", "parameters": {"type": "object", "properties": {"url": {"type": "array", "items": {"type": "string"}, "description": "The URL(s) of the webpage(s) to visit. Can be a single URL or an array of URLs."}, "goal": {"type": "string", "description": "The specific information goal for visiting webpage(s)."}}, "required": ["url", "goal"]}}}
{"type": "function", "function": {"name": "PythonInterpreter", "description": "Executes Python code in a sandboxed environment. To use this tool, you must follow this format:
1. The 'arguments' JSON object must be empty: {}.
2. The Python code to be executed must be placed immediately after the JSON block, enclosed within <code> and </code> tags.

IMPORTANT: Any output you want to see MUST be printed to standard output using the print() function.

Example of a correct call:
<tool_call>
{"name": "PythonInterpreter", "arguments": {}}
<code>
import numpy as np
# Your code here
print(f"The result is: {np.mean([1,2,3])}")
</code>
</tool_call>", "parameters": {"type": "object", "properties": {}, "required": []}}}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>

Current date: """


def today_date():
    """Get today's date in YYYY-MM-DD format."""
    return datetime.now().date().strftime("%Y-%m-%d")


def analyze_repetition_ngram(text: str, n: int = 30, threshold: float = 0.5):
    """
    Use N-grams to detect repetition in a string.

    Args:
        text (str): Input text to analyze.
        n (int): N-gram window size (default 10).
            - For long repetitive sequences, 10-20 is recommended.
        threshold (float): Distinct-N threshold (0~1).
            - Values below this indicate heavy repetition (default 0.5).

    Returns:
        bool: True if repetition is detected, False otherwise.
    """
    if not text or len(text) < n:
        print("text is too short, cannot analyze.")
        return False

    # 1. Generate N-grams (character-level sliding window).
    # List comprehension: slice from index i to i+n.
    ngrams = [text[i : i + n] for i in range(len(text) - n + 1)]

    total_count = len(ngrams)
    if total_count == 0:
        return False

    # 2. Count frequencies.
    ngram_counts = Counter(ngrams)
    unique_count = len(ngram_counts)

    # 3. Compute Distinct-N (unique count / total count).
    # Repetitive text is typically < 0.4.
    distinct_ratio = unique_count / total_count

    # 4. Determine repetition.
    is_repetitive = distinct_ratio < threshold

    return is_repetitive


def count_words(text: str) -> int:
    # Match segments that look like English words.
    # Rule: starts and ends with a letter, may contain letters, apostrophes, or hyphens.
    pattern = re.compile(r"[A-Za-z]+(?:['-][A-Za-z]+)*")
    words = pattern.findall(text)
    return len(words)


def build_text_completion_prompt(
    messages: list[dict], allow_special: bool = True
) -> str:
    """
    Build text completion prompt from messages list.
    Adapted from qwen_agent.utils.utils.build_text_completion_prompt

    Args:
        messages: List of message dictionaries with 'role' and 'content' keys
        allow_special: Whether to allow special tokens (for compatibility)

    Returns:
        Formatted prompt string
    """
    im_start = "<|im_start|>"
    im_end = "<|im_end|>"

    prompt_parts = []

    # Handle system message
    if messages and messages[0]["role"] == "system":
        sys_content = messages[0]["content"]
        prompt_parts.append(f"{im_start}system\n{sys_content}{im_end}")
        messages = messages[1:]

    # Ensure chat completes with assistant
    if messages and messages[-1]["role"] != "assistant":
        messages = messages + [{"role": "assistant", "content": ""}]

    # Format each message
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        prompt_parts.append(f"{im_start}{role}\n{content}{im_end}")

    return "\n".join(prompt_parts)


class MultiTurnReactAgent:
    """
    Multi-turn ReAct Agent adapted from Tongyi DeepResearch.

    This agent implements the core reasoning loop with tool calling capabilities,
    using AsyncOpenAI for model inference.
    """

    def __init__(
        self,
        client: AsyncOpenAI,
        model: str,
        sampling_params: dict = None,
        tools: dict = None,
        system_prompt: str | None = None,
        default_max_tries: int = 3,
        **kwargs,
    ):
        """
        Initialize the ReAct agent.

        Args:
            client: AsyncOpenAI client for model inference
            model: Model name to use
            sampling_params: Sampling parameters (temperature, top_p, max_tokens, etc.)
            tools: Dictionary of available tools {tool_name: tool_instance}
            system_prompt: Optional custom system prompt
        """
        self.client = client
        self.model = model
        self.sampling_params = sampling_params or {}
        self.tools = tools or {}
        self.system_prompt = system_prompt
        self.max_llm_calls = MAX_LLM_CALL_PER_RUN
        self.default_max_tries = default_max_tries

        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0

        self.max_prompt_tokens = 32768
        self.max_response_tokens = 8192
        self.max_context_tokens = self.max_prompt_tokens + self.max_response_tokens

    def sanity_check_output(self, content: str) -> bool:
        """Check if the model output contains the expected thinking structure."""
        return "<think>" in content and "</think>" in content

    def _extract_content(self, response) -> str:
        """Extract text content from an OpenAI chat completion response."""
        if response is None:
            return ""
        if hasattr(response, "choices") and response.choices:
            return response.choices[0].message.content or ""
        return ""

    async def call_server(
        self, messages: list[dict], max_tries: Optional[int] = None, **kwargs
    ):
        """Call OpenAI-compatible endpoint; assumes XML ReAct format."""
        try:
            resp = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                **self.sampling_params,
            )
            if resp.usage:
                self.total_prompt_tokens = resp.usage.prompt_tokens or 0
                self.total_completion_tokens = resp.usage.completion_tokens or 0
            return resp
        except Exception as exc:
            print(f"call_server failed: {exc}")
            raise

    def record_token_usage(self, response) -> None:
        """Record the latest prompt/completion token count from response."""
        if response and hasattr(response, "usage") and response.usage:
            self.total_prompt_tokens = response.usage.prompt_tokens or 0
            self.total_completion_tokens = response.usage.completion_tokens or 0

    def get_total_tokens_used(self) -> int:
        """Return the latest prompt + completion token usage."""
        return self.total_prompt_tokens + self.total_completion_tokens

    def _build_result(
        self,
        *,
        question: str,
        answer: str | None,
        messages: list[dict],
        prediction: str,
        termination: str,
        rounds: int,
        start_time: float,
    ) -> dict:
        """Assemble result payload with token usage metadata."""
        token_usage = {
            "prompt": self.total_prompt_tokens,
            "completion": self.total_completion_tokens,
            "max_prompt": self.max_prompt_tokens,
        }

        result = {
            "question": question,
            "answer": answer,
            "messages": messages,
            "prediction": prediction,
            "termination": termination,
            "rounds": rounds,
            "time_taken": time.time() - start_time,
            "token_usage": token_usage,
        }
        return result

    async def _run(
        self,
        question: str,
        answer: str = None,
        images: list = None,
        image_path: str = None,
        **kwargs,
    ) -> dict:
        """
        Main reasoning loop adapted from original DeepResearch.

        Args:
            question: The research question to answer
            answer: Ground truth answer (for evaluation)
            images: List of image data URLs (base64-encoded data: URIs)
            image_path: Local path to the first image (for crop_and_search)

        Returns:
            Dictionary with results including messages, prediction, and termination reason
        """
        start_time = time.time()

        system_prompt = (
            self.system_prompt or DEEPRESEARCH_SYSTEM_PROMPT
        ) + today_date()

        if images:
            # Build OpenAI multimodal content list
            content_parts = [{"type": "text", "text": question}]
            for img_url in images:
                content_parts.append(
                    {"type": "image_url", "image_url": {"url": img_url}}
                )
            user_message = {"role": "user", "content": content_parts}
        else:
            user_message = {"role": "user", "content": question}

        if not images:
            messages = [
                {
                    "role": "system",
                    "content": DEEPRESEARCH_SYSTEM_PROMPT_TEXT + today_date(),
                },
                user_message,
            ]
        else:
            messages = [
                {"role": "system", "content": system_prompt},
                user_message,
            ]

        num_llm_calls_available = self.max_llm_calls
        round = 0
        termination = None
        prediction = ""
        consecutive_bad_steps = 0
        repetition_count = 0

        while num_llm_calls_available > 0:
            round += 1
            num_llm_calls_available -= 1

            # Get model response
            try:
                response = await self.call_server(messages, **kwargs)
            except Exception as exc:
                prediction = "call_server failed"
                termination = "error"
                return self._build_result(
                    question=question,
                    answer=answer,
                    messages=messages,
                    prediction=prediction,
                    termination=termination,
                    rounds=round,
                    start_time=start_time,
                )

            self.record_token_usage(response)
            content = self._extract_content(response)

            if "<tool_call>" in content:
                if "python" in content.lower() and "<code>" in content:
                    pass
                elif '"name":' in content:
                    try:
                        tool_text = content.split("<tool_call>")[1].split(
                            "</tool_call>"
                        )[0]
                        tool_data = json5.loads(tool_text)
                        tool_name = tool_data.get("name", "Unknown")
                        if "arguments" in tool_data:
                            args_str = str(tool_data["arguments"])
                            pass
                        else:
                            pass
                    except Exception:
                        pass
                else:
                    pass

            # Clean up content if it contains tool_response
            if "<tool_response>" in content:
                pos = content.find("<tool_response>")
                content = content[:pos]

            # Only XML ReAct tool calls are supported.
            if "<tool_call>" in content and "</tool_call>" in content:
                # ReAct text format path
                assistant_message = {
                    "role": "assistant",
                    "content": content.strip(),
                    "step_error": False,
                }
                messages.append(assistant_message)
                tool_error = False

                tool_call_text = content.split("<tool_call>")[1].split("</tool_call>")[
                    0
                ]
                # Special handling for Python code (match original logic)
                if "python" in tool_call_text.lower():
                    try:
                        # Extract code from the original content (not just tool_call_text)
                        code_raw = (
                            content.split("<tool_call>")[1]
                            .split("</tool_call>")[0]
                            .split("<code>")[1]
                            .split("</code>")[0]
                            .strip()
                        )
                        result = await self.execute_python(code_raw)
                        if isinstance(result, str) and result.startswith(
                            (
                                "Python execution error:",
                                "PythonInterpreter tool not available",
                                "PythonInterpreter tool is not callable",
                            )
                        ):
                            tool_error = True
                    except Exception:
                        result = (
                            "[Python Interpreter Error]: Python code formatting error."
                        )
                        tool_error = True
                else:
                    try:
                        # Parse JSON tool call
                        tool_call = json5.loads(tool_call_text)
                        tool_name = tool_call.get("name", "")
                        tool_args = tool_call.get("arguments", {})
                        if tool_name == "crop_and_search":
                            tool_args["image_id"] = image_path
                        result = await self.custom_call_tool(tool_name, tool_args)
                    except Exception:
                        result = "[Json Parse Error]: Tool call is not a valid JSON."
                        tool_error = True

                if tool_error:
                    assistant_message["step_error"] = True

                # Add tool response in ReAct format
                tool_response = f"<tool_response>\n{result}\n</tool_response>"
                messages.append({"role": "user", "content": tool_response})

            # Check for final answer AFTER processing tools
            elif "<answer>" in content and "</answer>" in content:
                messages.append(
                    {
                        "role": "assistant",
                        "content": content.strip(),
                        "step_error": False,
                    }
                )
                prediction = content.split("<answer>")[1].split("</answer>")[0].strip()
                termination = "answer"
                consecutive_bad_steps = 0
                break

            # Priority 3: No tool call and answer, just reasoning or format error
            else:
                is_repetitive = analyze_repetition_ngram(content)
                is_overlong = count_words(content) > 2500
                if is_repetitive and is_overlong:
                    repetition_count += 1
                    print(f"Round {round}: Content repetition detected (count: {repetition_count}/{MAX_REPEAT_TURN})")

                    if repetition_count >= MAX_REPEAT_TURN:
                        final_instruction = {
                            "role": "user",
                            "content": f"Based on all the information above, please provide your best answer now in the format: <think>your final thinking</think>\n<answer>your answer</answer>"
                        }

                        messages.append(final_instruction)

                        print(f"Round {round}: Content repetition threshold reached, requesting final answer.")

                        try:
                            response = await self.call_server(messages)
                            final_content = self._extract_content(response)
                            messages.append(
                                {"role": "assistant", "content": final_content.strip()})

                            if "<answer>" in final_content and "</answer>" in final_content:
                                prediction = final_content.split(
                                    "<answer>")[1].split("</answer>")[0].strip()
                                termination = "answer"
                            else:
                                prediction = final_content.strip() if final_content.strip(
                                ) else "No answer found due to content repetition."
                                termination = "answer"
                        except Exception as exc:
                            prediction = "No answer found due to content repetition and model failure."
                            termination = "answer"

                        break
                    else:
                        # Repetition detected but below threshold, continue with format error handling
                        observation = f"Error: Content repetition detected. Invalid content format. Content must contain <tool_call> or <answer> tags. Let's try again."
                        messages.append(
                            {
                                "role": "assistant",
                                "content": content.strip(),
                                "step_error": True,
                            }
                        )
                        messages.append(
                            {"role": "user", "content": observation})
                else:
                    observation = "Error: Invalid content format. Content must contain <tool_call> or <answer> tags. Let's try again."
                    messages.append(
                        {
                            "role": "assistant",
                            "content": content.strip(),
                            "step_error": True,
                        }
                    )
                    messages.append({"role": "user", "content": observation})

            # Determine whether another round is feasible
            if num_llm_calls_available <= 0 and "<answer>" not in content:
                # Round limit reached, give model one final chance to answer
                final_instruction = {
                    "role": "user",
                    "content": f"You have reached the maximum number of reasoning rounds ({self.max_llm_calls}). Based on all the information gathered so far, please provide your best final answer now in the format: <think>your final thinking</think>\n<answer>your answer</answer>"
                }

                messages.append(final_instruction)

                print(f"Round {round}: Round limit reached, requesting final answer")

                try:
                    response = await self.call_server(messages)
                    final_content = self._extract_content(response)
                    messages.append(
                        {"role": "assistant", "content": final_content.strip()})

                    if "<answer>" in final_content and "</answer>" in final_content:
                        prediction = final_content.split(
                            "<answer>")[1].split("</answer>")[0].strip()
                        termination = "answer"
                    else:
                        prediction = final_content.strip() if final_content.strip(
                        ) else f"No answer found after {self.max_llm_calls} rounds."
                        termination = "answer"
                except Exception as exc:
                    prediction = f"No answer found after {self.max_llm_calls} rounds and model failure."
                    termination = f"round limit reached, model failed: {str(exc)}"

                return self._build_result(
                    question=question,
                    answer=answer,
                    messages=messages,
                    prediction=prediction,
                    termination=termination,
                    rounds=round,
                    start_time=start_time,
                )

        last_message_content = (
            messages[-1].get("content", "") if isinstance(messages[-1], dict) else ""
        )
        if isinstance(last_message_content, str) and "<answer>" in last_message_content:
            prediction = last_message_content.split("<answer>")[1].split("</answer>")[0]
            termination = "answer"
        else:
            prediction = "No answer found."
            termination = "answer not found"
            if num_llm_calls_available == 0:
                termination = "exceed available llm calls"

        result = self._build_result(
            question=question,
            answer=answer,
            messages=messages,
            prediction=prediction,
            termination=termination,
            rounds=round,
            start_time=start_time,
        )

        print("\nDeepResearch completed:")
        print(f"   Rounds: {round}")
        print(f"   Time: {result['time_taken']:.1f}s")
        print(f"   Termination: {termination}")
        print(
            "   Token usage: prompt={prompt}, completion={completion}".format(
                prompt=self.total_prompt_tokens,
                completion=self.total_completion_tokens,
            )
        )
        return result

    async def custom_call_tool(self, tool_name: str, tool_args: dict, **kwargs) -> str:
        """Execute tool calls with the available tools."""
        if tool_name in self.tools:
            try:
                if hasattr(self.tools[tool_name], "call"):
                    if asyncio.iscoroutinefunction(self.tools[tool_name].call):
                        result = await self.tools[tool_name].call(**tool_args)
                    else:
                        result = self.tools[tool_name].call(**tool_args)
                elif callable(self.tools[tool_name]):
                    result = self.tools[tool_name](**tool_args)
                else:
                    result = f"Tool {tool_name} is not callable"

                return str(result)

            except Exception as e:
                return f"Error calling tool {tool_name}: {e}"
        else:
            available_tools = list(self.tools.keys())
            return f"Tool {tool_name} not found. Available tools: {available_tools}"

    async def execute_python(self, code: str) -> str:
        """Execute Python code using the PythonInterpreter tool."""
        if "PythonInterpreter" in self.tools:
            try:
                tool = self.tools["PythonInterpreter"]
                if hasattr(tool, "call"):
                    if asyncio.iscoroutinefunction(tool.call):
                        result = await tool.call(code=code)
                    else:
                        result = tool.call(code=code)
                    return str(result)
                else:
                    return "PythonInterpreter tool is not callable"
            except Exception as e:
                return f"Python execution error: {e}"
        else:
            return "PythonInterpreter tool not available"

    def reset(self):
        """Reset the agent state for a new task."""
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0

    async def run(
        self,
        question: str,
        answer: str = None,
        images: list = None,
        image_path: str = None,
        **kwargs,
    ) -> dict:
        """
        Public interface for running the agent.

        Args:
            question: Research question to answer
            answer: Ground truth answer (optional, for evaluation)
            images: List of base64 data URI strings
            image_path: Local filesystem path to the first image (for crop_and_search)

        Returns:
            Result dictionary
        """
        self.reset()
        return await self._run(question, answer, images, image_path, **kwargs)


DeepResearchAgent = MultiTurnReactAgent
