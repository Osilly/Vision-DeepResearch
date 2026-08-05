"""
Vision-DeepResearch Example for slime Framework

This module provides multi-turn tool-calling capabilities for VLM training.
"""

from examples.vision_deepresearch.env import DeepResearchEnv, build_env
from examples.vision_deepresearch.tools import CropAndSearchTool, SearchTool, VisitTool

__all__ = [
    "DeepResearchEnv",
    "build_env",
    "SearchTool",
    "VisitTool",
    "CropAndSearchTool",
]

