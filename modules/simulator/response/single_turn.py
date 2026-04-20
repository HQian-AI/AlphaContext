from typing import List, Dict, Any
import random

class SingleTurnDataset:
    """A dataset wrapper for single-turn chat data without HuggingFace dependency."""

    def __init__(self, data: List[Dict[str, Any]], eval_ratio: float = 0.1, seed: int = 42):
        if not data:
            raise ValueError("Data cannot be empty")

        required_fields = {'prompt', 'completion'}
        self.fields = set(data[0].keys())

        if not required_fields.issubset(self.fields):
            missing = required_fields - self.fields
            raise ValueError(f"Missing required fields: {missing}")

        for i, entry in enumerate(data):
            if set(entry.keys()) != self.fields:
                raise ValueError(f"Entry {i} has inconsistent keys.")

        self.data = data
        self.eval_ratio = eval_ratio
        self.seed = seed


        random.seed(self.seed)
        eval_size = int(len(self.data) * self.eval_ratio)
        self.eval_indices = set(random.sample(range(len(self.data)), k=min(eval_size, len(self.data))))
        self.train_indices = set(range(len(self.data))) - self.eval_indices

    def __getitem__(self, idx):
        if isinstance(idx, str):
            if idx == "train":
                return [self.data[i] for i in self.train_indices]
            elif idx == "eval":
                return [self.data[i] for i in self.eval_indices]
            else:
                raise KeyError(f"Unknown split: {idx}")
        return self.data[idx]

    def __len__(self) -> int:
        return len(self.data)

    def get_splits_info(self) -> Dict[str, int]:
        return {
            'train': len(self.train_indices),
            'eval': len(self.eval_indices)
        }
