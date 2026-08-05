"""
Tool registry for Vision-DeepResearch multi-turn tool-calling rollout.

Exposes tool schemas used by the dataset and the chat template,
and provides factory functions for instantiating tool objects.
"""

from typing import Any


# ---------------------------------------------------------------------------
# Tool schemas (OpenAI-compatible, used in apply_chat_template)
# ---------------------------------------------------------------------------

TOOL_SEARCH = {
    "type": "function",
    "function": {
        "name": "search",
        "description": (
            "Performs batched web searches. Supply an array of query strings; "
            "the tool retrieves the top results for each query in one call. "
            "Use this to gather information from the web when you need up-to-date "
            "or factual information that you don't already know."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Array of query strings. Include multiple complementary search queries in a single call.",
                },
            },
            "required": ["query"],
        },
    },
}


TOOL_VISIT = {
    "type": "function",
    "function": {
        "name": "visit",
        "description": (
            "Visit one or more webpages and extract relevant information. "
            "Takes a URL (or array of URLs) and a goal description. "
            "Use this after search to gather detailed information from specific pages."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "url": {
                    "type": ["string", "array"],
                    "items": {"type": "string"},
                    "description": "The URL(s) of the webpage(s) to visit. Can be a single URL or an array of URLs.",
                },
                "goal": {
                    "type": "string",
                    "description": "The goal of the visit - what information you are looking for.",
                },
            },
            "required": ["url", "goal"],
        },
    },
}


TOOL_SELECT_CROP_SEARCH = {
    "type": "function",
    "function": {
        "name": "select_crop_search",
        "description": (
            "Visual search over multi-frame (video) inputs. "
            "Pick the most informative key frames from the input frames, then for each chosen frame "
            "draw a tight bounding box around the single most discriminative region (a logo, jersey, "
            "scoreboard, landmark, face, on-screen text, product, etc.). Each (frame, bbox) pair is "
            "cropped and visually searched in parallel, returning web evidence about the cropped content."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "image_idx": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "minItems": 1,
                    "description": (
                        "0-INDEXED frame indices. They map 1-to-1 to the `image_id: image_N` "
                        "labels in the user message — image_idx=N selects the frame labelled "
                        "`image_N`. For example image_idx=[0] picks `image_0` (the first frame); "
                        "image_idx=[0, 2] picks `image_0` and `image_2` (the 1st and 3rd frames). "
                        "If the user shows K frames, valid indices are 0..K-1; DO NOT use 1..K. "
                        "Choose only the frames that best reveal the target — avoid blurry, occluded, "
                        "or redundant frames. Must have the same length as bbox."
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
                        "One tight bounding box [x1,y1,x2,y2] per entry in image_idx, normalized to a "
                        "0-1000 scale (top-left = [0,0], bottom-right = [1000,1000]). Crop tightly around "
                        "the discriminative content — do not pass the whole frame. "
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
        },
    },
}


TOOL_CROP_AND_SEARCH = {
    "type": "function",
    "function": {
        "name": "crop_and_search",
        "description": (
            "Crop regions from an image and perform visual search to gather information. "
            "Takes an image_id (path or URL), bbox coordinates (single or multiple), and goal description. "
            "Use this to zoom into specific regions of an image and search for related visual information online."
        ),
        "parameters": {
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
                    "description": "Bounding box coordinates [x1,y1,x2,y2] or array of bboxes. Values in 0-1000 scale.",
                },
                "goal": {
                    "type": "string",
                    "description": "Description of what to search for in the cropped regions",
                },
            },
            "required": ["image_id", "bbox"],
        },
    },
}


def get_tools() -> list[dict[str, Any]]:
    """Return the list of tool schemas used in the dataset / chat template."""
    return [TOOL_SEARCH, TOOL_VISIT, TOOL_CROP_AND_SEARCH, TOOL_SELECT_CROP_SEARCH]


def get_tool_names() -> list[str]:
    """Return the names of all registered tools."""
    return [
        TOOL_SEARCH["function"]["name"],
        TOOL_VISIT["function"]["name"],
        TOOL_CROP_AND_SEARCH["function"]["name"],
        TOOL_SELECT_CROP_SEARCH["function"]["name"],
    ]


# ---------------------------------------------------------------------------
# Tool instances (lazily created in the env)
# ---------------------------------------------------------------------------

def make_search_tool():
    """Instantiate the search tool."""
    from .search_tool import SearchTool
    return SearchTool()


def make_visit_tool():
    """Instantiate the visit tool."""
    from .visit_tool import VisitTool
    return VisitTool()


def make_crop_and_search_tool():
    """Instantiate the crop_and_search tool."""
    from .crop_and_search_tool import CropAndSearchTool
    return CropAndSearchTool()


def make_select_crop_search_tool():
    """Instantiate the select_crop_search tool."""
    from .select_crop_search_tool import SelectCropSearchTool
    return SelectCropSearchTool()

