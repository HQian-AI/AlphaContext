import argparse
import json
import requests
import math
import os
import random
import re
from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np
from model_config import (
    MODEL_NAME_deepseek,
    CHAT_EP_deepseek,
    API_BASE_deepseek,
    API_KEY_deepseek,
    get_deepseek_headers,
)
from prompts.map_elites_prompts import (
    SYSTEM_PROMPT_FITNESS_EVALUATION,
    FITNESS_EVALUATION_PROMPT,
    SYSTEM_PROMPT_PHENOTYPE_EVALUATION,
    PHENOTYPE_EVALUATION_PROMPT,
    SYSTEM_PROMPT_MUTATION,
    MUTATION_PROMPT_TEMPLATE,
)

def is_valid_text(s: str, min_chars: int = 80) -> bool:

    if not isinstance(s, str):
        return False
    t = s.strip()
    if not t:
        return False

    if len(t) < min_chars:
        return False
    return True


class TextGenotype:
    def __init__(self, text: str, parts: Optional[List[str]] = None, original: Optional[dict] = None, fitness_scores: Optional[dict] = None):
        self.text = text

        self.parts = parts or []

        self.original = original

        self.fitness_scores = fitness_scores or {}

    def copy(self) -> "TextGenotype":
        return TextGenotype(self.text, parts=list(self.parts) if self.parts else None, original=self.original, fitness_scores=dict(self.fitness_scores) if self.fitness_scores else None)


def tokenize(text: str) -> List[str]:

    return [t.lower() for t in re.findall(r"\w+", text)]


class TextEnvironment:
    def __init__(
        self,
        initial_texts: List[str],
        rng: Optional[np.random.Generator] = None,
        use_model: bool = False,
        api_url: str = CHAT_EP_deepseek,
        api_key: str = API_KEY_deepseek,
        model: str = MODEL_NAME_deepseek,
    ):
        self.initial_texts = initial_texts

        self.behavior_ndim = 3

        low = np.array([0.0, 0.0, 0.0])
        high = np.array([1.0, 1.0, 1.0])
        self.behavior_space = (low, high)
        self.batch_size = 1
        self.rng = rng or np.random.default_rng()

        self.use_model = use_model
        self.api_url = api_url

        self.api_key = api_key 
        self.model = model

    def random(self) -> List[TextGenotype]:

        choices = self.rng.choice(len(self.initial_texts), size=self.batch_size, replace=True)
        out = []
        for i in choices:
            entry = self.initial_texts[int(i)]
            if isinstance(entry, TextGenotype):
                out.append(entry.copy())
            else:
                out.append(TextGenotype(str(entry)))
        return out

    def mutate(self, batch: List[TextGenotype], dimension_thresholds: Optional[dict] = None) -> List[TextGenotype]:

        if dimension_thresholds is None:
            dimension_thresholds = {}
        
        children = []
        for parent in batch:

            p = parent
            if not isinstance(p, TextGenotype):
                raise RuntimeError("Expected TextGenotype parent in mutate()")

            if not self.use_model:

                raise RuntimeError("mutate() called without model enabled; this code path is unsupported in model-only mode")


            relevance_target = float(self.rng.uniform(0.0, 1.0))
            evidence_target = float(self.rng.uniform(0.0, 1.0))
            stakeholder_target = float(self.rng.uniform(0.0, 1.0))
            
            dimension_hints = ""
            if p.fitness_scores:
                low_dimensions = []
                for dim, threshold in dimension_thresholds.items():
                    score = p.fitness_scores.get(dim, 0.0)
                    if score < threshold:
                        low_dimensions.append((dim, score, threshold))
                
                if low_dimensions:
                    dimension_hints = "\n\nOPTIMIZATION FOCUS:\n"
                    for dim, score, threshold in low_dimensions:
                        dimension_hints += f"- {dim.upper()}: Currently {score:.2f} (target {threshold:.2f}). "
                        if dim == "Coherence":
                            dimension_hints += "Improve logical consistency and smooth structure: ensure the problem context clearly anchors time, place, and scope, lets a single central creative challenge emerge naturally, and links background, events, and constraints without confusing jumps.\n"
                        elif dim == "Relevance":
                            dimension_hints += "Strengthen alignment with theme: ensure most sentences provide specific information, constraints, or stakeholder viewpoints that shape the creative problem space and support idea generation.\n"
                        elif dim == "Engagement":
                            dimension_hints += "Enhance motivation and positioning: present the creative challenge as meaningful and intriguing, and clearly position the reader as an active problem solver invited to explore possibilities.\n"
            
            # call model to produce mutated text
            mutated_text = model_mutate_via_api(
                p.text,
                relevance_target,
                evidence_target,
                stakeholder_target,
                api_url=self.api_url,
                api_key=self.api_key,
                model=self.model,
                dimension_hints=dimension_hints,
            )

            if not isinstance(mutated_text, str):
                continue

            text_out = mutated_text.strip()


            if not is_valid_text(text_out, min_chars=80):
                if getattr(self, "rng", None) is not None and self.rng.random() < 0.05:
                    print("[WARN] mutate produced invalid text, skipped. preview=", repr(text_out[:200]))
                continue

            children.append(TextGenotype(text_out, parts=[], original={}))

        return children

    def fitness(self, individual: TextGenotype) -> float:
        raise RuntimeError("Direct fitness() is deprecated; use evaluate() which calls the model to obtain both scores and phenotype.")

    def evaluate(self, individual: TextGenotype) -> Tuple[float, Optional[np.ndarray]]:

        if not self.use_model:
            # print("[DEBUG] Model evaluation skipped - use_model is False")
            return -np.inf, None
        if not is_valid_text(individual.text, min_chars=80):
            return -np.inf, None

        # Prefer separate model calls for fitness and phenotype to allow independent tuning.
        fitness = -np.inf
        # compute fitness from Coherence/Relevance/Engagement
        w = {"Coherence": 0.4, "Relevance": 0.3, "Engagement": 0.3}

        if self.use_model:
            try:
                fitness_scores = model_score_fitness_via_api(individual.text, api_url=self.api_url, api_key=self.api_key, model=self.model)
            except Exception:
                fitness_scores = {}
        else:
            fitness_scores = {}

        if all(k in fitness_scores for k in w.keys()):
            total = 0.0
            for k, weight in w.items():
                total += float(fitness_scores.get(k, 0.0)) * weight
            fitness = float(total)
            # Attach fitness scores to individual for later use in mutation
            individual.fitness_scores = fitness_scores
        else:
            fitness = -np.inf

        # phenotype scores (separate call)
        if self.use_model:
            try:
                phen_scores = model_score_phenotype_via_api(individual.text, api_url=self.api_url, api_key=self.api_key, model=self.model)
            except Exception:
                phen_scores = {}
        else:
            phen_scores = {}

        required_axes = ("student_relevance", "evidence_anchoring", "stakeholder_breadth")
        if not all(k in phen_scores for k in required_axes):
            return -np.inf, None

        sr = float(phen_scores.get("student_relevance", 0.0))
        ea = float(phen_scores.get("evidence_anchoring", 0.0))
        sb = float(phen_scores.get("stakeholder_breadth", 0.0))

        return fitness, np.array([sr, ea, sb], dtype=float)


def split_sentences(text: str) -> List[str]:
    # naive sentence splitter
    s = re.split(r'(?<=[.!?])\s+', text.strip())
    return [part for part in s if part]


def parts_from_text(text: str, max_parts: int = 5) -> List[str]:
    if not text:
        return []
    if "\n\n" in text:
        parts = [p.strip() for p in text.split("\n\n") if p.strip()]
        if len(parts) > max_parts:
            parts = parts[: max_parts - 1] + [" ".join(parts[max_parts - 1 :])]
        return parts
    sents = split_sentences(text)
    if not sents:
        return [text.strip()]
    n = len(sents)
    chunk_size = math.ceil(n / max_parts)
    parts = [" ".join(sents[i : i + chunk_size]).strip() for i in range(0, n, chunk_size)]
    if len(parts) > max_parts:
        parts = parts[: max_parts - 1] + [" ".join(parts[max_parts - 1 :])]
    return parts



def model_score_via_api(text: str, api_url: str, api_key: str, model: str) -> dict:
    fitness_scores = model_score_fitness_via_api(text, api_url=api_url, api_key=api_key, model=model)
    phen_scores = model_score_phenotype_via_api(text, api_url=api_url, api_key=api_key, model=model)
    merged = {}
    merged.update(fitness_scores)
    merged.update(phen_scores)
    return merged


def model_score_fitness_via_api(text: str, api_url: str, api_key: str, model: str) -> dict:
    """Request fitness-related scores (0..1 floats): Coherence, Relevance, Engagement."""
    chat_ep = api_url
    headers = get_deepseek_headers(api_key)

    system_msg = SYSTEM_PROMPT_FITNESS_EVALUATION

    user_prompt = FITNESS_EVALUATION_PROMPT + text

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.0,
        "max_tokens": 1500,
    }

    try:
        resp = requests.post(chat_ep, headers=headers, json=payload, timeout=120)
        resp.raise_for_status()
        out = resp.json()
    except requests.exceptions.RequestException as e:
        if hasattr(e, 'response') and e.response is not None:
            print(f"[DEBUG] Error response body: {e.response.text}")
        raise

    text_out = None
    if isinstance(out, dict):
        if "choices" in out and isinstance(out["choices"], list) and out["choices"]:
            first = out["choices"][0]
            if isinstance(first.get("message"), dict):
                text_out = first["message"].get("content")
            else:
                text_out = first.get("text") or first.get("message") or first.get("content")
        elif "result" in out:
            text_out = out["result"]
        else:
            text_out = json.dumps(out)
    else:
        text_out = str(out)


    try:
        start = text_out.find("{")
        end = text_out.rfind("}")
        if start != -1 and end != -1:
            j = json.loads(text_out[start:end+1])
            result = {}
            for k, v in j.items():
                try:
                    score_value = float(v)
                except Exception:
                    continue
                score_value = max(0.0, min(1.0, score_value))
                result[k] = score_value
            return {
                "Coherence": result.get("Coherence", result.get("coherence", 0.0)),
                "Relevance": result.get("Relevance", result.get("relevance", 0.0)),
                "Engagement": result.get("Engagement", result.get("engagement", 0.0)),
            }
    except Exception:
        pass

    return {}


def model_score_phenotype_via_api(text: str, api_url: str, api_key: str, model: str) -> dict:
    """Request phenotype-related scores (0..1 floats): student_relevance, evidence_anchoring, stakeholder_breadth."""
    chat_ep = api_url
    headers = get_deepseek_headers(api_key)

    system_msg = SYSTEM_PROMPT_PHENOTYPE_EVALUATION

    user_prompt = PHENOTYPE_EVALUATION_PROMPT + text

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.0,
        "max_tokens": 1024,
    }

    try:
        resp = requests.post(chat_ep, headers=headers, json=payload, timeout=120)
        resp.raise_for_status()
        out = resp.json()
    except requests.exceptions.RequestException as e:
        if hasattr(e, 'response') and e.response is not None:
            print(f"[DEBUG] Error response body: {e.response.text}")
        raise

    text_out = None
    if isinstance(out, dict):
        if "choices" in out and isinstance(out["choices"], list) and out["choices"]:
            first = out["choices"][0]
            if isinstance(first.get("message"), dict):
                text_out = first["message"].get("content")
            else:
                text_out = first.get("text") or first.get("message") or first.get("content")
        elif "result" in out:
            text_out = out["result"]
        else:
            text_out = json.dumps(out)
    else:
        text_out = str(out)

    try:
        start = text_out.find("{")
        end = text_out.rfind("}")
        if start != -1 and end != -1:
            j = json.loads(text_out[start:end+1])
            result = {}
            for k, v in j.items():
                try:
                    score_value = float(v)
                except Exception:
                    continue
                score_value = max(0.0, min(1.0, score_value))
                result[k.lower()] = score_value
            return {
                "student_relevance": result.get("student_relevance", 0.0),
                "evidence_anchoring": result.get("evidence_anchoring", 0.0),
                "stakeholder_breadth": result.get("stakeholder_breadth", 0.0),
            }
    except Exception:
        pass

    return {}


def model_mutate_via_api(txt: str, relevance_target: float, evidence_target: float, stakeholder_target: float, api_url: str, api_key: str, model: str, dimension_hints: str = "") -> str:
    chat_ep = api_url

    headers = get_deepseek_headers(api_key)

    system_msg = SYSTEM_PROMPT_MUTATION

    user_prompt = (
        MUTATION_PROMPT_TEMPLATE + txt + dimension_hints
    ).format(relevance_target, evidence_target, stakeholder_target)



    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.7,
        "max_tokens": 4096,
    }

    resp = requests.post(chat_ep, headers=headers, json=payload, timeout=120)
    resp.raise_for_status()
    out = resp.json()

    text_out = None
    if isinstance(out, dict):
        if "choices" in out and isinstance(out["choices"], list) and out["choices"]:
            first = out["choices"][0]
            if isinstance(first.get("message"), dict):
                text_out = first["message"].get("content")
            else:
                text_out = first.get("text") or first.get("message") or first.get("content")
        elif "result" in out:
            text_out = out["result"]
        else:
            text_out = json.dumps(out, ensure_ascii=False)
    else:
        text_out = str(out)

    # Since we now return the entire text, just return the text_out
    return text_out.strip()


class SimpleMap:
    """A minimal N-d grid-backed map specialised for this script."""

    def __init__(self, dims: Tuple[int, ...], fill_value: float = -np.inf):
        self.dims = tuple(int(d) for d in dims)
        self.fitnesses = np.full(self.dims, fill_value, dtype=float)
        self.genomes = np.full(self.dims, None, dtype=object)
        self.nonzero = np.zeros(self.dims, dtype=bool)

    def __getitem__(self, idx: Tuple[int, ...]):
        return self.fitnesses[idx]

    def __setitem__(self, idx: Tuple[int, ...], value):
        self.fitnesses[idx] = value

    def set_genome(self, idx: Tuple[int, ...], genome: TextGenotype):
        self.genomes[idx] = genome
        self.nonzero[idx] = True

    @property
    def latest(self):
        return self.fitnesses

    @property
    def niches_filled(self) -> int:
        return int(np.count_nonzero(self.nonzero))

    @property
    def qd_score(self) -> float:
        return float(np.nansum(self.fitnesses[self.nonzero]))


class MAPElitesText:
    def __init__(self, env: TextEnvironment, map_grid=(10, 10, 10), output_dir: str = "outputs/map_elites_text"):
        self.env = env
        self.grid = tuple(map_grid)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        # bins per dimension
        self.bins = [
            np.linspace(self.env.behavior_space[0][i], self.env.behavior_space[1][i], self.grid[i] + 1)[1:-1]
            for i in range(self.env.behavior_ndim)
        ]
        self.map = SimpleMap(self.grid)
        self.rng = env.rng
        # verbosity (set by caller via attribute)
        self.verbose = False

    def to_mapindex(self, phenotype: np.ndarray) -> Optional[Tuple[int, ...]]:
        if phenotype is None:
            return None
        idx = []
        for val, bins in zip(phenotype, self.bins):
            ix = int(np.digitize(val, bins))
            idx.append(ix)
        return tuple(idx)

    def random_selection(self) -> Tuple[int, int]:
        flat = np.flatnonzero(self.map.nonzero)
        if flat.size == 0:
            # choose random cell
            ix = tuple(int(x) for x in np.floor(self.rng.random(self.env.behavior_ndim) * np.array(self.grid)))
            return ix
        choice = int(self.rng.choice(flat))
        return tuple(np.unravel_index(choice, self.grid))

    def update_map(self, individuals: List[TextGenotype]):
        for ind in individuals:
            fit, phen = self.env.evaluate(ind)
            if not np.isfinite(fit) or phen is None:
                continue
            map_ix = self.to_mapindex(phen)
            if map_ix is None:
                continue
            # compare
            if fit > self.map.fitnesses[map_ix]:
                self.map.fitnesses[map_ix] = fit
                self.map.set_genome(map_ix, ind.copy())

    def search(self, init_steps: int = 50, total_steps: int = 500, idx: int = 1, 
               dimension_thresholds: Optional[dict] = None):
        if dimension_thresholds is None:
            dimension_thresholds = {"Coherence": 0.85, "Relevance": 0.85, "Engagement": 0.85}


        # initialize
        if self.map.niches_filled == 0:
            max_fitness = -np.inf
            max_genome = None
        else:
            max_fitness = np.nanmax(self.map.fitnesses[self.map.nonzero])
            max_idx = np.unravel_index(np.nanargmax(self.map.fitnesses), self.grid)
            max_genome = self.map.genomes[max_idx]

        # choose a reasonable reporting interval
        report_interval = max(1, min(10, total_steps // 20))
        for step in range(total_steps):
            if step < init_steps or self.map.niches_filled == 0:
                individuals = self.env.random()
            else:
                parents = []
                for _ in range(self.env.batch_size):
                    parents.append(self.map.genomes[self.random_selection()])
                individuals = self.env.mutate(parents, dimension_thresholds=dimension_thresholds)

            self.update_map(individuals)

            # periodic progress print to show activity (useful when model calls block)
            if self.verbose and (step % report_interval == 0 or step == total_steps - 1):
                niches = int(self.map.niches_filled)
                best_fit = None
                min_fit = None
                if self.map.nonzero.any():
                    best_idx = np.unravel_index(np.nanargmax(self.map.fitnesses), self.grid)
                    best_fit = float(self.map.fitnesses[best_idx])
                    min_fit = float(np.nanmin(self.map.fitnesses[self.map.nonzero]))
                # compute per-dimension stats (Coherence/Relevance/Engagement) across filled niches if available
                per_dim_stats = None
                if self.map.nonzero.any():
                    dims = ["Coherence", "Relevance", "Engagement"]
                    collected = {d: [] for d in dims}
                    for ix in np.ndindex(*self.grid):
                        if not self.map.nonzero[ix]:
                            continue
                        g = self.map.genomes[ix]
                        if g is None:
                            continue
                        fs = getattr(g, "fitness_scores", {}) or {}
                        for d in dims:
                            try:
                                collected[d].append(float(fs.get(d, float('nan'))))
                            except Exception:
                                collected[d].append(float('nan'))
                    per_dim_stats = {}
                    for d in dims:
                        arr = np.array([v for v in collected[d] if not np.isnan(v)], dtype=float)
                        if arr.size:
                            per_dim_stats[d] = {"min": float(np.nanmin(arr)), "mean": float(np.nanmean(arr))}
                        else:
                            per_dim_stats[d] = {"min": None, "mean": None}
                print(f"step {step+1}/{total_steps} - niches_filled={niches} - best_fitness={best_fit} - min_fitness={min_fit} - per_dim={per_dim_stats}")

            # update global best for logging
            if self.map.nonzero.any():
                best_idx = np.unravel_index(np.nanargmax(self.map.fitnesses), self.grid)
                best_fit = float(self.map.fitnesses[best_idx])
                best_genome = self.map.genomes[best_idx]
                if best_fit > max_fitness:
                    max_fitness = best_fit
                    max_genome = best_genome

        # after the run, save final bests only
        self.save_results(idx)
        
        # Return the minimum fitness among all filled cells (or -inf if no cells filled)
        if self.map.nonzero.any():
            min_fitness_in_map = float(np.nanmin(self.map.fitnesses[self.map.nonzero]))
            # compute per-dimension final stats for saving/logging
            dims = ["Coherence", "Relevance", "Engagement"]
            collected = {d: [] for d in dims}
            for ix in np.ndindex(*self.grid):
                if not self.map.nonzero[ix]:
                    continue
                g = self.map.genomes[ix]
                if g is None:
                    continue
                fs = getattr(g, "fitness_scores", {}) or {}
                for d in dims:
                    try:
                        collected[d].append(float(fs.get(d, float('nan'))))
                    except Exception:
                        collected[d].append(float('nan'))
            per_dim_final = {}
            for d in dims:
                arr = np.array([v for v in collected[d] if not np.isnan(v)], dtype=float)
                if arr.size:
                    per_dim_final[d] = {"min": float(np.nanmin(arr)), "mean": float(np.nanmean(arr))}
                else:
                    per_dim_final[d] = {"min": None, "mean": None}
            # print final per-dimension summary
            if self.verbose:
                print(f"MAP final per-dimension stats: {per_dim_final}")
        else:
            min_fitness_in_map = -np.inf
            per_dim_final = {d: {"min": None, "mean": None} for d in ["Coherence", "Relevance", "Engagement"]}
        
        return max_genome, min_fitness_in_map, per_dim_final

    def save_results(self, idx:int =1):
        contexts = []

        for ix in np.ndindex(*self.grid):
            if not self.map.nonzero[ix]:
                continue
            genome = self.map.genomes[ix]


            if genome is None or not is_valid_text(getattr(genome, "text", ""), min_chars=80):
                continue

            context = {"text": genome.text}

            contexts.append(context)

        out_ctx = {"contexts": contexts}
        out_path_ctx = self.output_dir / f"final_contexts_{idx}.json"
        with open(out_path_ctx, "w", encoding="utf-8") as f:
            json.dump(out_ctx, f, ensure_ascii=False, indent=2)

        out_legacy = {"contexts": contexts}
        out_path_legacy = self.output_dir / f"final_contexts_{idx}.json"
        with open(out_path_legacy, "w", encoding="utf-8") as f:
            json.dump(out_legacy, f, ensure_ascii=False, indent=2)

        meta = []
        for ix in np.ndindex(*self.grid):
            if not self.map.nonzero[ix]:
                continue
            genome = self.map.genomes[ix]
            if genome is None or not is_valid_text(getattr(genome, "text", ""), min_chars=80):
                continue

            fitness = float(self.map.fitnesses[ix])
            meta.append({
                "map_index": tuple(int(i) for i in ix),
                "fitness": fitness,
                "text_preview": str(genome.text)[:200],
            })

        meta_path_ctx = self.output_dir / f"final_contexts_meta_{idx}.json"
        with open(meta_path_ctx, "w", encoding="utf-8") as f:
            json.dump({"entries": meta}, f, ensure_ascii=False, indent=2)

        meta_path_legacy = self.output_dir / f"final_contexts_meta_{idx}.json"
        with open(meta_path_legacy, "w", encoding="utf-8") as f:
            json.dump({"entries": meta}, f, ensure_ascii=False, indent=2)


def load_initial_texts(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    out: List[TextGenotype] = []

    # Handle the common case where JSON has a top-level "contexts" or legacy "contexts" list
    if isinstance(data, dict):
        list_key = None
        if "contexts" in data and isinstance(data["contexts"], list):
            list_key = "contexts"
        elif "contexts" in data and isinstance(data["contexts"], list):
            list_key = "contexts"

        if list_key is not None:
            for item in data[list_key]:
                if isinstance(item, dict):
                    parts = []
                    for key in ("Anchor", "Scene Setting", "Characters & Interaction", "Conflict & Challenge", "Open Task"):
                        if key in item and isinstance(item[key], str) and item[key].strip():
                            parts.append(item[key].strip())
                    # fallback: if no structured parts, try common keys
                    if not parts:
                        for key in ("text", "context", "content"):
                            if key in item and isinstance(item[key], str) and item[key].strip():
                                parts = [item[key].strip()]
                                break
                    text = "\n\n".join(parts) if parts else json.dumps(item, ensure_ascii=False)
                    out.append(TextGenotype(text=text, parts=parts, original=item))
            return out

    # If top-level is a list of strings or dicts
    if isinstance(data, list):
        for item in data:
            if isinstance(item, str):
                out.append(TextGenotype(text=item, parts=[item], original=None))
            elif isinstance(item, dict):
                # try to extract text-like fields
                for key in ("text", "context", "content"):
                    if key in item and isinstance(item[key], str):
                        out.append(TextGenotype(text=item[key], parts=[item[key]], original=item))
                        break
                else:
                    out.append(TextGenotype(text=json.dumps(item, ensure_ascii=False), parts=[], original=item))
        return out

    # dict mapping id->context
    if isinstance(data, dict):
        for v in data.values():
            if isinstance(v, str):
                out.append(TextGenotype(text=v, parts=[v], original=None))
            elif isinstance(v, dict):
                for key in ("text", "context", "content"):
                    if key in v and isinstance(v[key], str):
                        out.append(TextGenotype(text=v[key], parts=[v[key]], original=v))
                        break
        return out

    return [TextGenotype(text=str(data), parts=[str(data)], original=None)]


def run_map_elites(
    input_file: str,
    output_dir: str,
    use_model: bool = False,
    total_steps: int = 200,
    init_steps: int = 50,
    grid_x: int = 10,
    grid_y: int = 10,
    grid_z: int = 10,
    seed: int = 42,
    api_url: str = CHAT_EP_deepseek,
    api_key: str = API_KEY_deepseek,
    model: str = MODEL_NAME_deepseek,
    verbose: bool = False,
    idx: int = 1,
    dimension_thresholds: Optional[dict] = None
):
    rng = np.random.default_rng(seed)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


    texts = load_initial_texts(input_file)
    if not texts:
        print(f" No initial texts found in {input_file}")
        return None, None


    env = TextEnvironment(
        texts,
        rng=rng,
        use_model=use_model,
        api_url=api_url,
        api_key=api_key,
        model=model
    )
    algo = MAPElitesText(
        env,
        map_grid=(grid_x, grid_y, grid_z),
        output_dir=str(output_dir)
    )
    algo.verbose = bool(verbose)


    best_genome, min_fitness_in_map, per_dim_final = algo.search(
        init_steps=init_steps,
        total_steps=total_steps,
        idx=idx,
        dimension_thresholds=dimension_thresholds
    )

    best_fit = None
    if algo.map.nonzero.any():
        best_idx = np.unravel_index(np.nanargmax(algo.map.fitnesses), algo.grid)
        best_fit = float(algo.map.fitnesses[best_idx])

    print(f" Done. best_fitness={best_fit} | min_fitness_in_map={min_fitness_in_map}")
    if per_dim_final is not None:
        print(f" Per-dimension final stats: {per_dim_final}")
    if best_genome:
        print(best_genome.text[:1000])


    return best_genome, best_fit, min_fitness_in_map, per_dim_final

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--init_steps", type=int, default=50)
    parser.add_argument("--total_steps", type=int, default=200)
    parser.add_argument("--grid_x", type=int, default=10)
    parser.add_argument("--grid_y", type=int, default=10)
    parser.add_argument("--grid_z", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--input", type=str, default="outputs/initial_contexts.json")
    parser.add_argument("--output", type=str, default="outputs/map_elites_text")
    parser.add_argument("--use_model", action="store_true", help="Enable remote model scoring")
    parser.add_argument("--api_url", type=str, default=CHAT_EP_deepseek)
    parser.add_argument("--api_key", type=str, default=API_KEY_deepseek, help="API key for DeepSeek model")
    parser.add_argument("--model", type=str, default="DeepSeek-V3.1", help="Model name to use for scoring")
    parser.add_argument("--verbose", action="store_true", help="Print periodic progress during search")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    texts = load_initial_texts(args.input)
    if not texts:
        print("No initial texts found in", args.input)
        return

    env = TextEnvironment(texts, rng=rng, use_model=args.use_model, api_url=args.api_url, api_key=args.api_key, model=args.model)
    algo = MAPElitesText(env, map_grid=(args.grid_x, args.grid_y, args.grid_z), output_dir=args.output)
    algo.verbose = bool(args.verbose)
    best_genome, min_fitness_in_map, per_dim_final = algo.search(
        init_steps=args.init_steps,
        total_steps=args.total_steps,
        idx=args.idx,
        dimension_thresholds=None
    )

    best_fit = None
    if algo.map.nonzero.any():
        best_idx = np.unravel_index(np.nanargmax(algo.map.fitnesses), algo.grid)
        best_fit = float(algo.map.fitnesses[best_idx])

    print(f" Done. best_fitness={best_fit} | min_fitness_in_map={min_fitness_in_map}")
    print(f"Per-dimension final stats: {per_dim_final}")
    if best_genome:
        print(best_genome.text[:1000])


if __name__ == "__main__":
    main()
