"""
SelectCropSearch tool for Vision-DeepResearch multi-turn tool-calling rollout.

Accepts a list of image indices and a corresponding list of bounding boxes
(one bbox per image), then searches all (image, bbox) pairs concurrently.
Inherits from CropAndSearchTool to reuse its internal machinery directly
without calling CropAndSearchTool.call().
"""

from __future__ import annotations

import asyncio
import os
from typing import List, Optional, Tuple, Union

from .crop_and_search_tool import CropAndSearchTool
from .shared import log_tool_event


class SelectCropSearchTool(CropAndSearchTool):
    """
    Search multiple images simultaneously, each with its own bounding box.

    ``image_idx`` and ``bbox`` are 1-to-1 paired: ``image_idx[i]`` is the
    position in the input image list and ``bbox[i]`` is the region to crop on
    that image.  Image index resolution (int → file path) is done in env.py
    before this tool's ``call()`` is invoked.

    Parameters accepted by ``call()``:
        image_paths (list[str]): Resolved file paths, one per entry in
            image_idx (env.py maps indices → paths before calling).
        bbox (list[list[int]]): One [x1,y1,x2,y2] bbox per image_path.
            Must have the same length as image_paths.  0-1000 scale.
        what (str): What to search for.
        where (str): Where the target is located in the image (optional hint).
    """

    def __init__(self):
        super().__init__()  # inherit all API-key setup and internal methods

        # Override tool identity
        self.name = "select_crop_search"
        self.description = (
            "Search one or more images simultaneously. "
            "Specify which images to search via their indices (image_idx) and "
            "provide one bounding box per image (bbox). "
            "Each (image, bbox) pair is processed concurrently."
        )
        self.parameters = {
            "type": "object",
            "properties": {
                "image_idx": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "minItems": 1,
                    "description": (
                        "0-INDEXED image indices. They map 1-to-1 to the `image_id: image_N` "
                        "labels in the user message — image_idx=N selects the image labelled "
                        "`image_N`. For example image_idx=[0] picks `image_0` (the first image); "
                        "image_idx=[0, 2] picks `image_0` and `image_2`. If the user shows K "
                        "images, valid indices are 0..K-1; DO NOT use 1..K. "
                        "Must have the same length as bbox."
                    ),
                },
                "bbox": {
                    "type": "array",
                    "items": {
                        "type": "array",
                        "items": {"type": "number"},
                        "minItems": 4,
                        "maxItems": 4,
                    },
                    "minItems": 1,
                    "description": (
                        "One bounding box [x1,y1,x2,y2] per entry in image_idx. "
                        "Coordinates are in 0-1000 scale. "
                        "Must have the same length as image_idx."
                    ),
                },
                "goal": {
                    "type": "string",
                    "description": (
                        "Describe the search target in two parts: "
                        "(1) what entity/object to search for, "
                        "(2) where it is located in the image and what specifically to look up about it. "
                        "Example: 'The brand logo in the top-left corner — identify the brand name and find its official website.'"
                    ),
                },
            },
            "required": ["image_idx", "bbox", "goal"],
        }
        self._json = {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def call(
        self,
        image_paths: List[str],
        bbox: List[List[int]],
        goal: str = "",
        **kwargs,
    ) -> str:
        """
        Crop and search all (image_path, bbox) pairs concurrently.

        ``image_paths`` and ``bbox`` are already paired and validated by
        env.py before this method is called.
        """
        cache_dir = self.image_crop_cache
        if cache_dir is None:
            return "[SelectCropSearch] IMAGE_CROP_CACHE must be provided."
        os.makedirs(cache_dir, exist_ok=True)

        tasks: List[asyncio.Task] = [
            asyncio.ensure_future(
                self._process_single_bbox(single_bbox, i, image_path, cache_dir, goal)
            )
            for i, (image_path, single_bbox) in enumerate(zip(image_paths, bbox))
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        all_results: List[str] = []
        for i, result in enumerate(results):
            header = f"[Image index {i} | path: {image_paths[i]} | bbox: {bbox[i]}]"
            if isinstance(result, Exception):
                log_tool_event(
                    "SelectCropSearch",
                    "TaskException",
                    f"image_{i} error={str(result)}",
                    level="ERROR",
                )
                all_results.append(f"{header}\nTask failed: {result}")
            elif isinstance(result, tuple) and len(result) >= 2:
                _, result_text, _ = result
                all_results.append(f"{header}\n{result_text}")
            else:
                log_tool_event(
                    "SelectCropSearch",
                    "InvalidResult",
                    f"image_{i} unexpected_type={type(result)}",
                    level="ERROR",
                )
                all_results.append(f"{header}\nInvalid result format")

        return "\n\n=======\n\n".join(all_results)
