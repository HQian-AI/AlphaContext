# creativity_dataset.py
from __future__ import annotations

import sys

import json
import os.path as osp
from typing import List, Dict, Any
import tiktoken

from modules.simulator.response.single_turn import SingleTurnDataset


def num_tokens_from_string(string: str, encoding_name: str = "cl100k_base") -> int:
    """Returns the number of (approximated) tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string, disallowed_special=()))
    return num_tokens
    

class Creativity(SingleTurnDataset):

    def __init__(
        self,
        root: str = "data/",
        *,
        min_tokens: int = 20,
        seed: int = 42,
    ):
        self.min_tokens = min_tokens
        self.seed = seed
        
        raw_ds = self._load_json_splits(root)
        processed = self._preprocess(raw_ds)
        super().__init__(processed, seed=seed)  # eval_ratio unused because we set `split`
    
    @staticmethod
    def _load_json_splits(root: str) -> Dict[str, Any]:
        splits = {}
        for name in ("train", "test"):
            path = osp.join(root, f"creativity/{name}.json")
            with open(path, "r", encoding="utf-8") as f:
                splits[name] = json.load(f)
        return splits

    # ------------------------------------------------------------------ #
    # helpers                                                            #
    # ------------------------------------------------------------------ #
    def _preprocess(self, raw_ds) -> List[Dict[str, Any]]:
        examples: List[Dict[str, Any]] = []
        for split_name, split_blob in raw_ds.items():
            for item in split_blob:
                title = item.get("title", "").strip()
                response = item.get("text", "").strip()
                if not title or not response:
                    continue
                tokens = num_tokens_from_string(response)
                if tokens < self.min_tokens:
                    continue
        
                prompt = (
                    f"Below is a text centered around the theme '{title}'. Please read carefully and think about the potential concrete consequences or challenges this event might bring based on the text content:\n\n"
                    f"{response}\n\nPlease identify the most relevant and significant potential core challenge in this contextual scenario."
                )

                examples.append(
                    {
                        # Required columns
                        "prompt": prompt,
                        "completion": "",
                        "split": split_name,
                        # Metadata columns
                        "title": title,
                        "num_tokens": tokens,
                    }
                )
        return examples
