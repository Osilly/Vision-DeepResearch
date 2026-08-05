"""
Base interaction environment for Vision-DeepResearch multi-turn tool-calling rollouts.

Defines the contract for interaction environments used in slime framework.
"""

from typing import Any


class BaseInteractionEnv:
    """
    Base class that defines the contract for interaction environments.

    Concrete envs must implement:
        reset()       -> (observation dict, reset_info dict)
        step(str)     -> (observation dict, done: bool, info dict)
        close()

    The base class also provides helpers for multi-modal observations.
    """

    def reset(self) -> tuple[dict[str, Any], dict[str, Any]]:
        raise NotImplementedError

    def step(self, response_text: str) -> tuple[dict[str, Any], bool, dict[str, Any]]:
        raise NotImplementedError

    def close(self):
        pass

    def format_observation(self, observation: dict[str, Any]) -> dict[str, Any]:
        """
        Convert an env observation into a chat message format for tokenization.

        Supports:
          - obs_str: plain text
          - multi_modal_data.images: list of image file paths or URLs
        """
        observation = observation or {}
        content = []

        multimodal = observation.get("multi_modal_data") or {}
        for key, items in multimodal.items():
            if key == "images":
                for img in items:
                    if img:
                        content.append({"type": "image", "image": img})

        text = observation.get("obs_str", "")
        if text:
            content.append({"type": "text", "text": text})

        return {"role": "user", "content": content}

