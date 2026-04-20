import os
import json
import re
import requests
from typing import Optional
from pathlib import Path
import ast
import spacy
import asyncio
from tqdm import tqdm
from modules.mcts import generate_MCTSContext
from modules.htp_outline import generate_outline
from modules.map_elites_context import run_map_elites
import concurrent.futures
from glob import glob
import metrics.creativity
from modules.simulator.src.reward import multiturn_aware_reward 
from model_config import (
    MODEL_NAME_deepseek,
    LITELLM_MODEL_NAME_deepseek,
    API_BASE_deepseek,
    CHAT_EP_deepseek,
    API_KEY_deepseek,
    get_deepseek_headers,
)

try:
    nlp = spacy.load("en_core_web_sm")
except Exception:
    try:
        from spacy.cli import download as spacy_download
        print("spaCy model en_core_web_sm not found, attempting to download...")
        spacy_download("en_core_web_sm")
        nlp = spacy.load("en_core_web_sm")
        print("spaCy model en_core_web_sm downloaded and loaded.")
    except Exception as e:
        print("Warning: spaCy en_core_web_sm not available and auto-download failed:", e)
        nlp = None


# -------------------------
# Utility Functions
# -------------------------
def extract_json_from_piece(piece_str: str):
    match = re.search(r"\{.*\}", piece_str, re.S)
    if not match:
        raise ValueError("No JSON found in piece field")
    raw = match.group().strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    try:
        return ast.literal_eval(raw)
    except Exception:
        pass
    fixed = re.sub(
        r"(\s*)([A-Za-z_][A-Za-z0-9_]*)(\s*):",
        r'\1"\2"\3:',
        raw
    )
    try:
        return json.loads(fixed)
    except Exception as e:
        raise ValueError(f"Failed to parse JSON even after repair attempt. Raw content: {raw[:200]}... Error: {e}")


# -------------------------
# Context Class 
# -------------------------
class Context:
    def __init__(self, anchor, scene_setting, characters_and_interaction, conflict_and_challenge, open_task, score=0.0):
        self.anchor = anchor
        self.scene_setting = scene_setting
        self.characters_and_interaction = characters_and_interaction
        self.conflict_and_challenge = conflict_and_challenge
        self.open_task = open_task
        self.score = score

    def to_dict(self):
        return {
            "Anchor": self.anchor,
            "Scene Setting": self.scene_setting,
            "Characters & Interaction": self.characters_and_interaction,
            "Conflict & Challenge": self.conflict_and_challenge,
            "Open Task": self.open_task,
            "Score": self.score
        }

    @classmethod
    def from_piece(cls, piece_str):
        data = extract_json_from_piece(piece_str)
        return cls(
            data.get("Anchor", ""),
            data.get("Scene Setting", ""),
            data.get("Characters & Interaction", []),
            data.get("Conflict & Challenge", ""),
            data.get("Open Task", ""),
        )


def convert_to_collab_format(input_file, output_file):
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)
        contexts = data.get("contexts", data.get("contexts", [])) if isinstance(data, dict) else data
    output_data = []
    for i, context in enumerate(contexts, 1):
        raw_text = ""
        if isinstance(context, dict):
            raw_text = (context.get("text") or context.get("raw_context_text") or "").strip()
        else:
            raw_text = str(context).strip()
        if raw_text:
            output_data.append({
                "title": (context.get("title") if isinstance(context, dict) else "") or f"Context {i}",
                "text": raw_text
            })
            continue

        Anchor = context.get("Anchor", "") if isinstance(context, dict) else ""
        Scene_Setting = context.get("Scene Setting", "") if isinstance(context, dict) else ""
        Characters_and_Interaction = context.get("Characters & Interaction", []) if isinstance(context, dict) else []
        Conflict_and_Challenge = context.get("Conflict & Challenge", "") if isinstance(context, dict) else ""
        Open_Task = context.get("Open Task", "") if isinstance(context, dict) else ""
        characters_lines = []
        if isinstance(Characters_and_Interaction, list):
            for ch in Characters_and_Interaction:
                if isinstance(ch, dict):
                    category = ch.get("Category", "")
                    desc = ch.get("Description", "")
                    line = f"{category}: {desc}".strip(": ")
                else:
                    line = str(ch)
                if line.strip():
                    characters_lines.append(line.strip())
        else:
            if str(Characters_and_Interaction).strip():
                characters_lines.append(str(Characters_and_Interaction).strip())

        text = "\n".join([p for p in [
            Anchor, Scene_Setting, "\n".join(characters_lines), Conflict_and_Challenge, Open_Task
        ] if p and str(p).strip()]).strip()

        output_data.append({
            "title": Anchor if Anchor else f"Context {i}",
            "text": text
        })

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    return output_file


def score_contexts_with_multiple_personas(context_file, base_personalities=None, per_persona_count=10,
                                          output_subdir="outputs/multiturn_data_personas"):
    if base_personalities is None:
        base_personalities = ['quiet', 'talkative', 'default']
    personas = []
    for base in base_personalities:
        for i in range(1, int(per_persona_count) + 1):
            personas.append(f"{base}_{i}")

    print(f" Scoring with {len(personas)} virtual personas")
    os.makedirs(output_subdir, exist_ok=True)
    assess_context = os.path.join("outputs", "creativity", "assess_context.json")
    os.makedirs(os.path.dirname(assess_context), exist_ok=True)
    convert_to_collab_format(context_file, assess_context)

    with open(assess_context, "r", encoding="utf-8") as f:
        collab_data = json.load(f)

    model_kwargs = {
        "model": LITELLM_MODEL_NAME_deepseek,
        "api_base": API_BASE_deepseek,
        "api_key": API_KEY_deepseek,
        "model_cost": {
            "input_cost_per_token": 0,
            "output_cost_per_token": 0,
            "max_input_tokens": 32768,
            "max_output_tokens": 8192,
        }
    }
    all_persona_scores = []

    def _score_sample_for_persona(persona_name: str, item: dict):
        try:
            base_personality = persona_name.rsplit('_', 1)[0]
            user_gen_kwargs = dict(model_kwargs)
            user_gen_kwargs["prompt_variant"] = base_personality
            text = item.get("text", "")
            reward_dict = multiturn_aware_reward(
                task_desc="Evaluate the creativity of the context.",
                single_turn_prompt=text,
                single_turn_completion="",
                metric_names=["creativity"],
                user_generation_kwargs=user_gen_kwargs,
                assistant_generation_kwargs=model_kwargs,
                reward_generation_kwargs=model_kwargs,
                metric_weights=[1.0],
                chat_history=[],
                max_new_turns=1,
                num_samples=1,
                max_workers=20,
                max_metric_workers=16,
                return_details=False,
            )
            creativity_vals = reward_dict.get("creativity", [0.0])
            creativity_score = sum(creativity_vals) / len(creativity_vals) if creativity_vals else 0.0
            return creativity_score
        except Exception as e:
            import traceback
            print(f"[ERROR] _score_sample_for_persona failed for {persona_name}: {e}")
            traceback.print_exc()
            return 0.0

    def _score_persona(persona_name: str):
        try:
            scores = [0.0] * len(collab_data)

            with concurrent.futures.ThreadPoolExecutor(max_workers=8) as sample_executor:
                future_to_idx = {
                    sample_executor.submit(_score_sample_for_persona, persona_name, item): i
                    for i, item in enumerate(collab_data)
                }
                for fut in concurrent.futures.as_completed(future_to_idx):
                    i = future_to_idx[fut]
                    try:
                        scores[i] = float(fut.result() or 0.0)
                    except Exception:
                        scores[i] = 0.0

            persona_avg = sum(scores) / len(scores) if scores else 0.0
            return {"persona": persona_name, "scores": scores, "avg": persona_avg}
        except Exception:
            import traceback
            traceback.print_exc()
            return {"persona": persona_name, "scores": [0.0] * len(collab_data), "avg": 0.0}

    max_persona_workers = min(15, len(personas))
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_persona_workers) as executor:
        future_to_persona = {executor.submit(_score_persona, p): p for p in personas}
        completed = 0
        total = len(future_to_persona)
        for fut in concurrent.futures.as_completed(future_to_persona):
            p = future_to_persona[fut]
            try:
                res = fut.result()
            except Exception:
                import traceback
                traceback.print_exc()
                res = {"persona": p, "scores": [], "avg": 0.0}
            all_persona_scores.append(res)
            completed += 1
            print(f"    ▶ Completed {completed}/{total}: {p} avg={res.get('avg', 0.0):.4f}")

    # --- per-item average ---
    num_items = len(collab_data)
    per_item_avg = [0.0] * num_items
    if all_persona_scores and num_items > 0:
        for i in range(num_items):
            vals = []
            for ps in all_persona_scores:
                scs = ps.get("scores") or []
                if i < len(scs):
                    vals.append(float(scs[i]))
            per_item_avg[i] = (sum(vals) / len(vals)) if vals else 0.0

    all_scores = []
    for ps in all_persona_scores:
        all_scores.extend(ps["scores"])
    grand_avg = sum(all_scores) / len(all_scores) if all_scores else 0.0
    print(f"\n Grand average creativity score (across {len(personas)} personas): {grand_avg:.4f}")
    return {
        "grand_average": grand_avg,
        "per_item_average": per_item_avg,
        "per_persona": all_persona_scores,
        "all_scores": all_scores
    }


# -------------------------
# score_contexts
# -------------------------
def score_contexts(context_file):
    output_dir = "outputs/multiturn_data"
    os.makedirs(output_dir, exist_ok=True)
    assess_context = os.path.join("outputs", "creativity", "assess_context.json")
    os.makedirs(os.path.dirname(assess_context), exist_ok=True)
    convert_to_collab_format(context_file, assess_context)
    print(" Running scoring with 30 virtual personas ")
    multi_persona_result = score_contexts_with_multiple_personas(
        context_file=context_file,
        base_personalities=['quiet', 'talkative', 'default'],
        per_persona_count=10,
        output_subdir=output_dir
    )
    output_path = os.path.join(output_dir, "final_creative_scores.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(multi_persona_result, f, ensure_ascii=False, indent=2)

    print(f" Saved multi-persona scoring results to: {output_path}")
    per_item_scores = multi_persona_result.get("per_item_average") or []
    fallback_score = float(multi_persona_result.get("grand_average", 0.0))
    combined = []
    try:
        with open(context_file, "r", encoding="utf-8") as f:
            orig_data = json.load(f)
            if isinstance(orig_data, dict):
                orig_list = orig_data.get("contexts", orig_data.get("contexts", []))
            else:
                orig_list = orig_data
    except Exception:
        orig_list = []
    for i, orig in enumerate(orig_list):
        s = dict(orig) if isinstance(orig, dict) else {"text": str(orig)}
        sc = per_item_scores[i] if i < len(per_item_scores) else fallback_score
        try:
            sc = float(sc)
        except Exception:
            sc = fallback_score
        s["score"] = sc
        s["Score"] = sc
        combined.append(s)
    try:
        _scores = [c.get("score", 0.0) for c in combined]
        print(f"DEBUG per-item score range: {min(_scores):.4f} -> {max(_scores):.4f}")
    except Exception:
        pass

    return {"contexts": combined}


def get_average_score(scored_data):
    contexts = scored_data.get("contexts", scored_data.get("contexts", []))
    scores = [s.get("score", 0) for s in contexts]
    return sum(scores) / len(scores) if scores else 0

def assemble_full_context(context: dict, model_kwargs: Optional[dict] = None) -> str:
    Anchor = context.get("Anchor", "")
    Scene_Setting = context.get("Scene Setting", "")
    Characters_and_Interaction = context.get("Characters & Interaction", context.get("Characters_and_Interaction", ""))
    Conflict_and_Challenge = context.get("Conflict & Challenge", "")
    Open_Task = context.get("Open Task", context.get("Open_Task", ""))
    chars_texts = []
    if isinstance(Characters_and_Interaction, list):
        for ch in Characters_and_Interaction:
            if isinstance(ch, dict):
                cat = ch.get("Category", "")
                desc = ch.get("Description", "")
                if cat or desc:
                    chars_texts.append(f"{cat}: {desc}".strip(": "))
                else:
                    chars_texts.append(json.dumps(ch, ensure_ascii=False))
            else:
                chars_texts.append(str(ch))
    elif isinstance(Characters_and_Interaction, dict):
        cat = Characters_and_Interaction.get("Category", "")
        desc = Characters_and_Interaction.get("Description", "")
        if cat or desc:
            chars_texts.append(f"{cat}: {desc}".strip(": "))
        else:
            chars_texts.append(json.dumps(Characters_and_Interaction, ensure_ascii=False))
    else:
        if Characters_and_Interaction:
            chars_texts.append(str(Characters_and_Interaction))
    chars_text = "\n".join([t for t in chars_texts if t])

    narrative_blocks = [
        ("Anchor", Anchor),
        ("Scene Setting", Scene_Setting),
        ("Characters & Interaction", chars_text),
        ("Conflict & Challenge", Conflict_and_Challenge),
    ]
    if not any(b[1] and str(b[1]).strip() for b in narrative_blocks) and not (Open_Task and str(Open_Task).strip()):
        if context.get("text"):
            return str(context.get("text"))
    prompt_parts = [
        "You are tasked with connecting these context segments by adding transition sentences between them.",
        "Rules:",
        "1. Keep ALL original content exactly as is",
        "2. Add 1-2 connecting sentences BETWEEN segments to create smooth flow",
        "3. Focus on cause-effect, time progression, or perspective shifts in your transitions",
        "Example of what we want:",
        "Original segments:",
        "Segment1: The city's lights flickered in the rain.",
        "Segment2: Sarah typed frantically at her computer.",
        "With transition:",
        "The city's lights flickered in the rain. Through one of those rain-streaked windows, in a downtown office that never slept, Sarah typed frantically at her computer.",
        "\nNow, add similar transitions between these context segments while keeping their original content intact:\n"
    ]
    narrative_text = []
    for name, content in narrative_blocks:
        if content and str(content).strip():
            narrative_text.append(f"### {name} ###\n{content}")

    prompt = "\n".join(prompt_parts + narrative_text)
    stitched = None
    try:
        stitched = _call_local_model_raw(prompt, model_kwargs=model_kwargs, timeout=120)
    except Exception:
        stitched = None

    if stitched and isinstance(stitched, str) and stitched.strip():
        stitched_text = stitched.strip()
        stitched_text = re.sub(r'###\s*(?:Anchor|Scene Setting|Characters & Interaction|Conflict & Challenge)\s*###\s*',
                               '', stitched_text)
    else:
        stitched_text = "\n\n".join([c for _, c in narrative_blocks if c and str(c).strip()])

    if Open_Task and str(Open_Task).strip():
        stitched_text = f"{stitched_text}\n\n{str(Open_Task).strip()}"
        
    return stitched_text


def _call_local_model_raw(prompt: str, model_kwargs: dict, timeout: int = 120) -> Optional[str]:
    api_url = model_kwargs.get("api_url") or model_kwargs.get("api_base")
    model_name = model_kwargs.get("model")

    if not api_url or not model_name:
        print("_call_local_model_raw: api_base or model name not found in model_kwargs.")
        return None

    if not api_url.rstrip("/").endswith(("/chat/completions", "/completions")):
        api_url = api_url.rstrip("/") + "/chat/completions"

    headers = get_deepseek_headers(model_kwargs.get("api_key"))

    payload = {
        "model": model_name,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.0,
        "max_tokens": 4096,
    }

    try:
        response = requests.post(api_url, headers=headers, json=payload, timeout=timeout)
        response.raise_for_status()
        data = response.json()

        if "choices" in data and data["choices"]:
            message = data["choices"][0].get("message", {})
            content = message.get("content")
            if content:
                return content.strip()

        print(f" _call_local_model_raw: Unexpected API response format: {data}")
        return None

    except requests.exceptions.RequestException as e:
        print(f"Local model call failed in _call_local_model_raw: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Error response body: {e.response.text}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred in _call_local_model_raw: {e}")
        return None

def produce_final_context(scored_path: str, idx: int, input_title: Optional[str] = None) -> Optional[str]:
    try:
        with open(scored_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f" Failed to read scoring file: {e}")
        return None

    contexts = data.get("contexts", [])
    if not contexts:
        print(" No context data available")
        return None

    def score_of(s):
        return float(s.get("score", s.get("Score", 0.0)))

    contexts_sorted = sorted(contexts, key=score_of, reverse=True)
    best = contexts_sorted[0]

    final_text = best.get("text", "")
    
    if not final_text.strip():
        final_text = best.get("raw_context_text", "")

    final_text = re.sub(r'[\{\}\[\]"\']', '', final_text).strip()
    
    out_dir = Path("outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"final_context_{idx}.json"
    

    output_dict = {
        "title": input_title or best.get("Anchor", f"Context {idx}"),
        "text": final_text
    }
    
    with open(json_path, "w", encoding="utf-8") as jf:
        json.dump(output_dict, jf, ensure_ascii=False, indent=2)

    print(f" Final context saved: {json_path}")
    return final_text
def save_all_final_contexts(output_path="outputs/final_contexts/final_contexts_all.json"):
    ctx_files = glob("outputs/final_context_*.json")

    def extract_idx(path):
        m = re.search(r"final_context_(\d+)\.json", path)
        return int(m.group(1)) if m else 99999

    ctx_files = sorted(ctx_files, key=extract_idx)
    out_list = []
    for idx, file in enumerate(ctx_files, 1):
        with open(file, "r", encoding="utf-8") as f:
            data = json.load(f)
        title = data.get("title", f"Context {idx}")
        text = data.get("text", "")
        # Check if text is valid
        if isinstance(text, (dict, list)) or not text.strip() or text in ["0", "0.0"]:
            print(f"Not valid text!!!!!!")
        else:
            text = re.sub(r'[\"\'\{\}\[\]]', '', text)  
            text = re.sub(r'score:\s*\d+\.\d+|raw_context_text:', '', text)  
            text = text.strip()
        out_list.append({
                "id": idx,
                "title": title,
                "method": "AlphaContext",
                "text": text  
            })

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(out_list, f, ensure_ascii=False, indent=2)
    print(f" Merged all contexts and saved to: {output_path}")


def main_loop():
    default_input = "input/test_dataset.json"

    if not os.path.exists(default_input):
        print(f"Input file does not exist: {default_input}, creating test input")
        test_input = [{"title": "AI Partner", "theme": "Human-AI Collaboration"}]
        os.makedirs(os.path.dirname(default_input), exist_ok=True)
        with open(default_input, "w", encoding="utf-8") as f:
            json.dump(test_input, f, ensure_ascii=False, indent=2)

    with open(default_input, "r", encoding="utf-8") as f:
        query_data_list = json.load(f)

    start_idx = 1

    for idx, query_data in enumerate(tqdm(query_data_list, desc="Generating creative contexts"), start=1):
        if idx < start_idx:
            print(f"\n Skipping topic {idx}, proceed to next")
            continue

        avg_score = 0
        generation = 1
        input_file = None
        # 1. Outline Generation
        if generate_outline is None:
            print(f"\n Topic {idx}: Outline generation module not imported, using raw data")
        else:
            print(f"\n Topic {idx}: Generating outline (Title: {query_data.get('title')})")
            generate_outline(query_data, idx)

        # 2. MCTS Context Generation
        if generate_MCTSContext is None:
            print(f" Topic {idx}: MCTS module not imported")
        else:
            num_contexts = 1
            outfile = asyncio.run(generate_MCTSContext(idx, num_contexts))
            input_file = outfile

        # Convert structured contexts from MCTS to plain text format
        with open(input_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        src_list = data.get("contexts", data.get("contexts", []))
        original_open_task = ""
        if src_list:
            for item in src_list:
                val = item.get("Open Task", item.get("Open_Task", ""))
                if val:
                    original_open_task = str(val).strip()
                    break
        assembled_list = []
        for d in src_list:
            assembled_text = assemble_full_context(d, model_kwargs=None)
            assembled_list.append({"text": assembled_text, "score": d.get("Score", 0.0)})        

        final_text_content = assembled_text
        if original_open_task and original_open_task not in final_text_content:
                final_text_content = f"{final_text_content}\n\n{original_open_task}"
        assembled_list.append({
                "text": assembled_text, 
                "score": d.get("Score", 0.0),
            })
        assembled_file = f"outputs/assembled_for_map_{idx}.json"
        with open(assembled_file, "w", encoding="utf-8") as f:
            json.dump({"contexts": assembled_list}, f, ensure_ascii=False, indent=2)
        input_file = assembled_file  # Update input_file to converted text format file
        
        # 3. MAP-Elites Evolution
        print(f"\n Topic {idx}: Starting MAP-Elites evolution")

        while avg_score < 80 and generation <= 3:
            print(f"\n=== Topic {idx} - Generation {generation} ===")

            # Read parent contexts
            with open(input_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            map_output_dir = Path("outputs/map_elites_text_model_run")
            map_output_dir.mkdir(parents=True, exist_ok=True)

            # Call MAP-Elites
            MAP_FITNESS_THRESHOLD = 0.9
            DIMENSION_THRESHOLDS = {"Coherence": 0.9, "Relevance": 0.95, "Engagement": 0.9}
            MAX_MAP_ATTEMPTS = 3
            attempt = 0
            best_genome = None
            min_fitness = -1.0
            per_dim_final = None
            per_dim_ok = False

            while attempt < MAX_MAP_ATTEMPTS and (min_fitness < MAP_FITNESS_THRESHOLD or not per_dim_ok):
                attempt += 1
                print(f"  ▶ MAP-Elites attempt {attempt}/{MAX_MAP_ATTEMPTS}")
                try:
                    best_genome, best_fit, min_fitness, per_dim_final = run_map_elites(
                        input_file=input_file,
                        output_dir=str(map_output_dir),
                        use_model=True,
                        total_steps=8,
                        init_steps=2,
                        grid_x=3, grid_y=3, grid_z=3,
                        api_url=CHAT_EP_deepseek,
                        api_key=API_KEY_deepseek,
                        model=MODEL_NAME_deepseek,
                        verbose=True,
                        idx=idx,
                        dimension_thresholds=DIMENSION_THRESHOLDS,
                    )
                except Exception as e:
                    print(f"  MAP-Elites attempt {attempt} failed: {e}")
                    if attempt >= MAX_MAP_ATTEMPTS:
                        print("  Reached maximum attempt limit, stopping evolution")
                        break
                    else:
                        continue

                print(f"  → Attempt {attempt} completed, min_fitness: {min_fitness}")
                if min_fitness is None:
                    min_fitness = -1.0

                # Check dimension thresholds
                per_dim_ok = True
                if per_dim_final and isinstance(per_dim_final, dict):
                    for dim, thresh in DIMENSION_THRESHOLDS.items():
                        v = per_dim_final.get(dim, {}).get("min")
                        try:
                            if v is None or float(v) < float(thresh):
                                per_dim_ok = False
                                break
                        except Exception:
                            per_dim_ok = False
                            break
                else:
                    per_dim_ok = False

                if min_fitness >= MAP_FITNESS_THRESHOLD and per_dim_ok:
                    print(f"   Threshold requirements met, stopping attempts")
                    break

            if best_genome is None and min_fitness < 0:
                print("MAP-Elites generated no valid results, stopping evolution")
                break

            # MAP-Elites output file
            next_gen_file = map_output_dir / f"final_contexts_{idx}.json"

            # Validate output file
            if not next_gen_file.exists():
                print(f" MAP-Elites did not generate file: {next_gen_file}, stopping evolution")
                break

            try:
                with open(next_gen_file, "r", encoding="utf-8") as nf:
                    js = json.load(nf)
                    if not (
                            (isinstance(js, dict) and ("contexts" in js or "contexts" in js) and isinstance(
                                js.get("contexts", js.get("contexts")), list))
                            or isinstance(js, list)
                    ):
                        print(f" Abnormal file format, stopping evolution")
                        break
            except Exception as e:
                print(f" Failed to read MAP-Elites output file: {e}, stopping evolution")
                break

            # Score contexts
            scored_data = score_contexts(next_gen_file)
            avg_score = get_average_score(scored_data)
            print(f"Current average score: {avg_score:.2f}")

            # Stop if target score reached
            if avg_score >= 80:
                print(" Evolution completed, average score reached target!")
                break

            # Update input file for next generation
            input_file = str(next_gen_file)
            generation += 1

            if generation > 3:
                print(" Reached maximum generation limit (3 generations), stopping evolution.")

        # Save scoring results
        current_dir = Path(__file__).parent
        score_save_dir = current_dir / "outputs"
        score_save_dir.mkdir(parents=True, exist_ok=True)
        score_out_path = score_save_dir / f"final_contexts_scored_{idx}.json"
        try:
            with open(score_out_path, "w", encoding="utf-8") as f:
                json.dump(scored_data, f, ensure_ascii=False, indent=2)
            print(f" Saved scoring file: {score_out_path}")
        except Exception as e:
            print(f"Failed to save scoring file: {e}")

        # Generate final coherent context
        local_model_kwargs = {
            "model": MODEL_NAME_deepseek,
            "api_base": API_BASE_deepseek,
            "api_key": API_KEY_deepseek,
            "model_cost": {
                "input_cost_per_token": 0,
                "output_cost_per_token": 0,
                "max_input_tokens": 32768,
                "max_output_tokens": 8192,
            }
        }
        produce_final_context(str(score_out_path), idx, input_title=query_data.get("title"))

    # Merge and save all contexts
    try:
        save_all_final_contexts()
    except Exception as e:
        print(f" Failed to merge and save all contexts: {e}")


if __name__ == "__main__":
    main_loop()