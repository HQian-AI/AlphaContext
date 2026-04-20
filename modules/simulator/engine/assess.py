import hashlib
import json
import os
import os.path as osp
from tqdm import tqdm
from typing import Tuple
import warnings
import concurrent.futures
from modules.simulator.src.synthetic import generate_multiturn_dataset
from single_turn_ds import datasets_info
from metrics import *


def compute_hash(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()

def get_unique_filename(output_dir, filename_base, ext=".json"):
    os.makedirs(output_dir, exist_ok=True)
    filepath = osp.join(output_dir, filename_base + ext)
    if not osp.exists(filepath):
        return filepath

    i = 1
    while True:
        new_filename = f"{filename_base}_{i}{ext}"
        new_path = osp.join(output_dir, new_filename)
        if not osp.exists(new_path):
            return new_path
        i += 1




def build_dataset(
    dataset_name: str,
    metric_names: List[str],
    user_generation_kwargs: Dict[str, Any],
    assistant_generation_kwargs: Dict[str, Any],
    reward_generation_kwargs: Dict[str, Any],
    metric_weights: List[float] = None,
    proact_prompt_ratio: float = 0.5,
    add_system_prompt_ratio: float = 0.0,
    num_candidate_responses: int = 1,
    max_total_turns: int = 3,
    max_new_turns: int = 2,
    num_samples: int = 1,
    train_size: int = 500,
    max_workers: int = 40,
    max_metric_workers: int = 40,
    output_dir: str = "outputs/multiturn_data",
    resume: bool = False,
    max_gen_workers: int = 8,
 ) -> Tuple[str, List[Dict[str, Any]]]:



    dataset_cls = datasets_info[dataset_name]["class"]
    task_desc = datasets_info[dataset_name]["task_desc"]

    dataset = dataset_cls()
    train = dataset["train"][:train_size] if train_size > 0 else dataset["train"]


    os.makedirs(output_dir, exist_ok=True)
    output_path = get_unique_filename(output_dir, f"{dataset_name}_multiturn", ext=".json")

    data_list: List[Dict[str, Any]] = []
    seen_prompt_hashes = set()

    if osp.exists(output_path):
        if resume:
            with open(output_path, "r", encoding="utf-8") as f:
                data_list = json.load(f)
            seen_prompt_hashes = {compute_hash(ex["prompt"]) for ex in data_list}
        else:
            warnings.warn(
                f"Output file {output_path} already exists. "
                f"Use resume=True to append new data.",
                UserWarning,
            )
            return

    pending_examples = [
        ex for ex in train if compute_hash(ex["prompt"]) not in seen_prompt_hashes
    ]

    if not pending_examples:
        print("No new examples to generate (all seen).")
        return

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_gen_workers) as executor:
        future_to_hash = {}
        for example in pending_examples:
            prompt_hash = compute_hash(example["prompt"])
            future = executor.submit(
                generate_multiturn_dataset,
                task_desc=task_desc,
                single_turn_prompt=example["prompt"],
                single_turn_completion=example["completion"],
                single_turn_metadata=example.get("metadata", {}),
                metric_names=metric_names,
                user_generation_kwargs=user_generation_kwargs,
                assistant_generation_kwargs=assistant_generation_kwargs,
                reward_generation_kwargs=reward_generation_kwargs,
                metric_weights=metric_weights,
                proact_prompt_ratio=proact_prompt_ratio,
                num_candidate_responses=num_candidate_responses,
                max_total_turns=max_total_turns,
                max_new_turns=max_new_turns,
                num_samples=num_samples,
                max_workers=min(num_samples, 4),
                max_metric_workers=max_metric_workers,
                add_system_prompt_ratio=add_system_prompt_ratio,
            )
            future_to_hash[future] = prompt_hash

        for future in tqdm(
            concurrent.futures.as_completed(future_to_hash),
            total=len(future_to_hash),
            desc="Generating multi-turn conversations",
        ):
            prompt_hash = future_to_hash[future]
            try:
                multiturn_data = future.result()
            except Exception as e:
                print(f" Error generating for prompt {prompt_hash}: {e}")
                continue

            if multiturn_data is None:
                continue

            data_list.append(multiturn_data)
            seen_prompt_hashes.add(prompt_hash)


            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(data_list, f, indent=2)

    print(f" Dataset Complete: {output_path}")

    return output_path, data_list
