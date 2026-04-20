import json
import glob
import os
import random
import numpy as np
from typing import Any, Dict, List, Optional, Sequence, Union

from prompts import SYSTEM_PROMPT

_REQUIRED = {
    "prompt",
    "completion",
    "conv_id",
    "score",
    "single_turn_prompt",
    "single_turn_completion",
    "single_turn_metadata",
}

import logging
logger = logging.getLogger(__name__)


def _load_local_json_any(path: str):
    if os.path.isfile(path):
        if path.endswith(".jsonl"):
            with open(path, "r", encoding="utf-8") as f:
                return [json.loads(line) for line in f if line.strip()]
        elif path.endswith(".json"):
            with open(path, "r", encoding="utf-8") as f:
                obj = json.load(f)
            if isinstance(obj, list):
                return obj
            if isinstance(obj, dict) and "data" in obj and isinstance(obj["data"], list):
                return obj["data"]
            if isinstance(obj, dict):  # {"train": [...], "eval": [...]} 等
                rows = []
                for v in obj.values():
                    if isinstance(v, list):
                        rows.extend(v)
                return rows
            raise ValueError(f"Unsupported JSON structure in {path}")
        else:
            raise ValueError(f"Unsupported file extension: {path}")

    if os.path.isdir(path):
        files = sorted(
            glob.glob(os.path.join(path, "*.json")) +
            glob.glob(os.path.join(path, "*.jsonl"))
        )
        if not files:
            raise FileNotFoundError(f"No .json/.jsonl under dir: {path}")
        rows = []
        for fp in files:
            rows.extend(_load_local_json_any(fp))
        return rows

    raise FileNotFoundError(path)


def _uniform_split(
    records: List[Dict[str, Any]],
    *,
    eval_ratio: float = 0.1,
    n_eval: Optional[int] = None,
    seed: int = 42,
):
    """Python list train/eval"""
    if not records:
        return {"train": [], "eval": []}
    k = n_eval if n_eval is not None else int(eval_ratio * len(records))
    k = min(k, len(records))
    random.seed(seed)
    eval_idx = set(random.sample(range(len(records)), k=k))
    train = [r for i, r in enumerate(records) if i not in eval_idx]
    evals = [r for i, r in enumerate(records) if i in eval_idx]
    return {"train": train, "eval": evals}


class MultiturnDataset:
    def __init__(
        self,
        data_or_local_dir_or_nested: Union[List[Dict[str, Any]], str],
        *,
        seed: int = 42,
        add_system_prompt: bool = True,
    ):
        self.seed = seed
        self.sys_msg = [{"role": "system", "content": SYSTEM_PROMPT}] if add_system_prompt else []

        # 加载
        src = data_or_local_dir_or_nested
        if isinstance(src, list):
            raw_list = src
        elif isinstance(src, str) and os.path.exists(src):
            raw_list = _load_local_json_any(src)
        else:
            raise ValueError("Only local list/json/jsonl/dir supported (HF repo not supported).")

        if not raw_list:
            raise ValueError("Loaded dataset is empty.")

        if isinstance(raw_list[0], dict) and "turns" in raw_list[0]:
            self.data = self._flatten_nested(raw_list)
        else:
            if not _REQUIRED.issubset(raw_list[0]):
                missing = _REQUIRED - set(raw_list[0])
                raise ValueError(f"Missing required keys: {missing}")
            for row in raw_list:
                if not isinstance(row["prompt"], Sequence):
                    raise TypeError("`prompt` must be a list of messages.")
                row.setdefault("turn_id", len(row["prompt"]))
            self.data = raw_list

    def _flatten_nested(self, nested: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        flat = []
        for base_conv_id, convo in enumerate(nested):
            for key in {"single_turn_prompt", "single_turn_completion", "single_turn_metadata", "turns"}:
                if key not in convo:
                    raise ValueError(f"Missing key '{key}' in nested conversation.")
            st_prompt = convo["single_turn_prompt"]
            st_completion = convo["single_turn_completion"]
            st_metadata = convo["single_turn_metadata"]
            for turn in convo["turns"]:
                prompt_msgs = turn["prompt"]
                turn_id = len(prompt_msgs)
                for resp in turn["responses"]:
                    flat.append(
                        {
                            "prompt": prompt_msgs,
                            "completion": resp["completion"],
                            "conv_id": base_conv_id,
                            "score": resp["score"],
                            "single_turn_prompt": st_prompt,
                            "single_turn_completion": st_completion,
                            "single_turn_metadata": st_metadata,
                            "turn_id": turn_id,
                        }
                    )
        return flat

    def to_sft_dataset(self, *, n_eval=None, eval_ratio=0.0):
        best_examples = {}
        for row in self.data:
            cid = row["conv_id"]
            prev = best_examples.get(cid)
            if prev is None or row["turn_id"] > prev["turn_id"] or (
                row["turn_id"] == prev["turn_id"] and row["score"] > prev["score"]
            ):
                best_examples[cid] = row
        serialized = [
            self.sys_msg + r["prompt"] + [{"role": "assistant", "content": r["completion"]}]
            for r in best_examples.values()
        ]
        return _uniform_split([{"messages": m} for m in serialized], eval_ratio=eval_ratio, n_eval=n_eval, seed=self.seed)

    def to_dpo_dataset(self, *, minimum_gap=0.0, n_eval=None, eval_ratio=0.0):
        grouped = {}
        for r in self.data:
            grouped.setdefault((r["conv_id"], r["turn_id"]), []).append(r)
        pairs = []
        for items in grouped.values():
            if len(items) < 2:
                continue
            items = sorted(items, key=lambda r: r["score"], reverse=True)
            if items[0]["score"] - items[-1]["score"] < minimum_gap:
                continue
            pairs.append({
                "prompt": self.sys_msg + items[0]["prompt"],
                "chosen": items[0]["completion"],
                "rejected": items[-1]["completion"],
                "score_chosen": items[0]["score"],
                "score_rejected": items[-1]["score"],
            })
        return _uniform_split(pairs, eval_ratio=eval_ratio, n_eval=n_eval, seed=self.seed)

    def to_inputs_dataset(self, *, n_eval=None, eval_ratio=0.0):
        unique = {}
        for r in self.data:
            key = (r["conv_id"], r["turn_id"])
            if key not in unique:
                unique[key] = r
        records = [{
            "prompt": self.sys_msg + row["prompt"],
            "single_turn_prompt": row["single_turn_prompt"],
            "single_turn_completion": row["single_turn_completion"],
            "single_turn_metadata": row["single_turn_metadata"],
        } for row in unique.values()]
        return _uniform_split(records, eval_ratio=eval_ratio, n_eval=n_eval, seed=self.seed)

    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.data[idx]
