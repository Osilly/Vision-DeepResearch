"""
DeepResearch Tools - Production-ready implementations

This module provides tool implementations for the DeepResearch agent, with real
functionality ported from Tongyi's original implementations where possible.

Now supports both:
- ReAct text format (for gpt-4o, Claude, etc.)
- OpenAI native function calling (for o3, o3-mini, etc.)
"""

from vision_deepresearch_async_workflow.tools.crop_and_search_tool import (
    CropAndSearchTool,
)
from vision_deepresearch_async_workflow.tools.python_interpreter_tool import (
    PythonInterpreterTool,
)
from vision_deepresearch_async_workflow.tools.search_tool import SearchTool
from vision_deepresearch_async_workflow.tools.shared import DeepResearchTool
from vision_deepresearch_async_workflow.tools.visit_tool import VisitTool

# Tool registry
DEEPRESEARCH_TOOLS = {
    "search": SearchTool(),
    "visit": VisitTool(),
    "PythonInterpreter": PythonInterpreterTool(),
    "crop_and_search": CropAndSearchTool(),  # Enable if PIL, requests, oss2 are available
}


def get_tool(name: str) -> DeepResearchTool:
    """Get a tool by name."""
    return DEEPRESEARCH_TOOLS.get(name)


def get_all_tools() -> dict[str, DeepResearchTool]:
    """Get all available tools."""
    return DEEPRESEARCH_TOOLS.copy()


__all__ = [
    "DeepResearchTool",
    "SearchTool",
    "VisitTool",
    "PythonInterpreterTool",
    "CropAndSearchTool",
    "DEEPRESEARCH_TOOLS",
    "get_tool",
    "get_all_tools",
]
