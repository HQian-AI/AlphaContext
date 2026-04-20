import re
SAFE_NAME_RE = re.compile(r'[^0-9A-Za-z\u4e00-\u9fa5._-]+')

import argparse
import asyncio
import json
import math
import random
from collections import defaultdict
import aiohttp
import requests
from typing import Callable, List, Any, Awaitable,Dict,  Optional,Tuple
import time
from pathlib import Path
import hashlib
from model_config import (
    MODEL_NAME_deepseek,
    CHAT_EP_deepseek,
    API_BASE_deepseek,
    API_KEY_deepseek,
    get_deepseek_headers,
)
from prompts.mcts_prompts import PART_TEMPLATES, SYSTEM_PROMPT_GENERATION, EVALUATION_PROMPT
# Default hosted model configuration (used when no model_kwargs passed)
DEFAULT_HOSTED_MODEL_KWARGS = {
    "model": MODEL_NAME_deepseek,
    "api_base": API_BASE_deepseek,
    "api_key": API_KEY_deepseek,
    "headers": get_deepseek_headers(),
    "model_cost": {
        "input_cost_per_token": 0,
        "output_cost_per_token": 0,
        "max_input_tokens": 32768,
        "max_output_tokens": 8192,
    }
}


class AsyncMCTSNode:
    def __init__(self, state: List[str], parent: Optional['AsyncMCTSNode'] = None, action: Optional[str] = None):
        self.state = state
        self.parent = parent
        self.action = action
        self.children: Dict[str, 'AsyncMCTSNode'] = {}
        self.visit_count = 0
        self.total_value = 0.0
        self.untried_actions: Optional[List[str]] = None

    def value(self) -> float:
        return 0.0 if self.visit_count == 0 else (self.total_value / self.visit_count)


class AsyncMCTS:
    def __init__(
        self,
        generate_actions_fn: Callable[[List[str], int], Awaitable[Any]],
        rollout_fn: Callable[[List[str], str, int], Awaitable[Any]],
        evaluate_state_fn: Callable[[str, str, bool, Optional[str]], Awaitable[Any]], # 注意：现在接收 context 参数
        c_puct: float = 0.01,
        num_simulations: int = 10,
        num_expansion_actions: int = 4,
        max_rollout_steps: int = 6,
        module_start_idx: int = 0,
    ):
        self.generate_actions_fn = generate_actions_fn
        self.rollout_fn = rollout_fn
        self.evaluate_state_fn = evaluate_state_fn
        self.c_puct = c_puct
        self.num_simulations = num_simulations
        self.num_expansion_actions = num_expansion_actions
        self.max_rollout_steps = max_rollout_steps
        self.module_start_idx = module_start_idx

    async def search(self, root_state: List[str], target_depth: int):
        root = AsyncMCTSNode(root_state)
        max_depth_recorded = 0
        for i in range(self.num_simulations):
            node = root
            current_depth = 0
        
            # 1. AsyncMCTS.search()  Selection
            while node.children and current_depth < target_depth:  
                if node.untried_actions:
                    break
                node = self._select_child(node)
                current_depth += 1

            # 2. Expansion
            if current_depth < target_depth:
                if node.untried_actions is None:
                    #  Debug code: View model generated candidates
                    candidates = await self.generate_actions_fn(node.state, self.num_expansion_actions)
                    if not candidates:
                        print(f" [Simulation {i}] Depth {current_depth}: Model returned empty candidates, this branch is dead!")
                    node.untried_actions = candidates
                
                if node.untried_actions:
                    action = node.untried_actions.pop()
                    next_state = list(node.state) + [action]
                    child = AsyncMCTSNode(next_state, parent=node, action=action)
                    node.children[action] = child
                    node = child
                    current_depth += 1
            max_depth_recorded = max(max_depth_recorded, current_depth)
            # 3. Simulation & Evaluation
            history = "\n".join(node.parent.state) if node.parent else ""
            current_sentence = node.action if node.action else ""
            
            full_text = history + "\n" + current_sentence if history else current_sentence
            module_text = "\n".join(node.state[self.module_start_idx:]) + "\n" + current_sentence
            
            _, _, _, _, value = await self.evaluate_state_fn(full_text, module_text, use_model=True)

            if value < 0.5:
                value = await self.rollout_fn(node.state, current_sentence, self.max_rollout_steps)

            # 4. Backpropagation
            self._backpropagate(node, value)

        #  5. Path Extraction summary
        print(f"\n --- MCTS Search Summary ---")
        print(f"Target depth: {target_depth} | Max depth reached: {max_depth_recorded} | Simulations: {self.num_simulations}")
        
        best_path = []
        curr = root
        for d in range(target_depth):
            if not curr.children:
                print(f" Path extraction stopped at layer {d}: Current node has no children (Children Count = 0)")
                break

            child_info = {act[:20] + "...": child.visit_count for act, child in curr.children.items()}
            print(f"Layer {d+1} branch visit counts: {child_info}")

            best_action, next_node = max(curr.children.items(), key=lambda kv: kv[1].visit_count)
            best_path.append(best_action)
            curr = next_node
        
        return best_path
        
    def _select_child(self, node: AsyncMCTSNode) -> AsyncMCTSNode:
        # Classic UCT: choose child maximizing Q + c * sqrt(2 * ln N(parent) / N(child))
        # Use the parent's visit count N(s) (node.visit_count) as required by UCT.
        parent_visits = max(1, node.visit_count)
        best_score = -float('inf')
        best_child = None
        for action, child in node.children.items():
            if child.visit_count == 0:
                return child
            q = child.value()
            u = self.c_puct * math.sqrt(2.0 * math.log(parent_visits) / child.visit_count)
            score = q + u
            if score > best_score:
                best_score = score
                best_child = child
        assert best_child is not None
        return best_child

    def _backpropagate(self, node: AsyncMCTSNode, value: float):
        cur = node
        while cur is not None:
            cur.visit_count += 1
            cur.total_value += value
            cur = cur.parent


class MCTSContext:
    def __init__(self, model: Optional[str] = None, model_kwargs: Optional[Dict[str, Any]] = None,
                outline_text: Optional[str] = None,
                few_shot_example: Optional[str] = None):  
        self.model = model
        self.model_kwargs = model_kwargs or DEFAULT_HOSTED_MODEL_KWARGS
        self.w_scaff = 0.4
        self.w_img = 0.3
        self.w_coh = 0.3
        self.debug = True
        self.few_shot_example = few_shot_example
        self.outline_text = outline_text
        self.eval_cache = {}
        self.part_hints = self._parse_outline(outline_text) if outline_text else {}

    def _parse_outline(self, outline_text: str) -> Dict[str, Any]:
            lines = outline_text.split('\n')
            tree = {}
            stack = []
            for line in lines:
                if not line.strip(): continue
                level = line.count('<Tab>')
                content = line.replace('<Tab>', '').strip()
                
                node_name = content[1:-1] if (content.startswith('[') and content.endswith(']')) else content
                node = {'name': node_name, 'children': []}
                
                while stack and stack[-1]['level'] >= level:
                    stack.pop()
                if stack:
                    stack[-1]['node']['children'].append(node)
                else:
                    tree[node_name] = node
                stack.append({'node': node, 'level': level})

            def _build_hierarchy(node: Dict) -> Any:
                if not node['children']:
                    return node['name']
                child_res = {}
                for child in node['children']:
                    child_res[child['name']] = _build_hierarchy(child)
                if all(isinstance(v, str) and k == v for k, v in child_res.items()):
                    return list(child_res.keys())
                return child_res

            part_hints = {}
            for name, node in tree.items():
                if name in PART_TEMPLATES:
                    if node['children']:
                        part_content = {}
                        for sub_node in node['children']:
                            part_content[sub_node['name']] = _build_hierarchy(sub_node)
                        part_hints[name] = part_content
                    else:
                        part_hints[name] = {}
            
            return part_hints

    def _clamp01(self, value: float) -> float:
        return max(0.0, min(1.0, value))

    def _json_extract_object(self, text: str) -> Dict:
        try:
            start = text.find('{')
            end = text.rfind('}')
            if start != -1 and end != -1:
                return json.loads(text[start:end+1])
        except:
            pass
        return {}
    async def _call_hosted_model(self, prompt: str, temperature: float = 0.8,
                             max_tokens: int = 2048, system_prompt: str = None) -> str:
        url = CHAT_EP_deepseek
        messages = []
        final_system_prompt = system_prompt or ""
        if self.few_shot_example:
            final_system_prompt += "\n\nHere's a high-quality example of a complete context structure. Learn from its narrative flow and style:\n" + self.few_shot_example
        
        if final_system_prompt:
            messages.append({"role": "system", "content": final_system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": MODEL_NAME_deepseek,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False
        }
        headers = get_deepseek_headers()

        loop = asyncio.get_event_loop()

        def _do_request():
            try:
                resp = requests.post(url, json=payload, headers=headers, timeout=60)
                resp.raise_for_status()
                data = resp.json()
                # try common shapes
                if isinstance(data, dict) and "choices" in data and data["choices"]:
                    c0 = data["choices"][0]
                    if isinstance(c0, dict):
                        if "message" in c0 and isinstance(c0["message"], dict):
                            return c0["message"].get("content", "")
                        if "text" in c0:
                            return c0.get("text", "")
                # fallback to raw text
                return resp.text
            except requests.exceptions.RequestException as e:
                print("LLM Error:", e)
                return ""

        return await loop.run_in_executor(None, _do_request)



    async def _generate_candidates(self, title: str, theme: str, part_name: str, state: List[str], k: int, temperature: float = 0.8) -> List[str]:
        context = "\n".join(state)
        tpl = PART_TEMPLATES.get(part_name)
        # Respect a small number of retries for parsing/validation failures
        max_attempts = 3
        attempt = 0
        final_candidates: List[str] = []

        max_chars = int(tpl.get('max_chars', 10000))
        model_max_tokens = int(tpl.get('max_tokens', self.model_kwargs.get('model_cost', {}).get('max_output_tokens', 8092)))
        tem = float(tpl.get('temperature', temperature))
        # allow asking for up to max 8 candidates to increase quality when needed
        requested_k = int(min(max(k, tpl.get('candidates', k)), 8))

        while attempt < max_attempts:
            attempt += 1
            prompt_text = tpl['prompt'].format(
                title=title,
                theme=theme,
                context=context,
                k=requested_k,
                max_chars=max_chars
            )
            
            if part_name in self.part_hints:
                structured_hints = json.dumps(self.part_hints[part_name], indent=2, ensure_ascii=False)
                prompt_text += f"\n\nOutline hints for {part_name} (Follow this structure):\n{structured_hints}"
                prompt_text += "\n\nEnsure the generated content adheres to the hierarchy and themes defined in the outline hints above."

            if self.model_kwargs is None:
                raise RuntimeError("No hosted model configured: please set model_kwargs for the hosted endpoint in the script")
            system_p = SYSTEM_PROMPT_GENERATION.format(title=title, theme=theme)
            raw = await self._call_hosted_model(prompt_text, temperature=tem, max_tokens=model_max_tokens, system_prompt=system_p)
            if getattr(self, 'debug', False):
                print(f"\n[DEBUG] raw model response for part={part_name}, attempt={attempt}:")
                print(raw)

            # Try robust JSON extraction first
            candidates = []
            try:
                start = raw.find('[')
                end = raw.rfind(']')
                if start != -1 and end != -1 and end > start:
                    arr = json.loads(raw[start:end+1])
                    print(f"arr is :::: {arr}\n")
                    if isinstance(arr, list):
                        candidates = [str(x).strip() for x in arr]
            except Exception:
                candidates = []

            # If JSON parse failed, attempt simple line splitting
            if not candidates:
                line_candidates = []
                for ln in raw.splitlines():
                    ln = ln.strip()
                    if not ln:
                        continue
                    if ln.startswith('[') and ln.endswith(']'):
                        try:
                            arr = json.loads(ln)
                            if isinstance(arr, list):
                                line_candidates.extend([str(x).strip() for x in arr])
                        except Exception:
                            continue
                if line_candidates:
                    candidates = line_candidates
                    print(f" merged line-wise candidates: {candidates}\n")
            processed: List[str] = []
            for c in candidates:
                # clean surrounding quotes/noise
                c = c.strip('"\'')
                c = re.sub(r'^\s*["\'\(\[]+', '', c)
                c = re.sub(r'["\'\)\]]+\s*$', '', c)
                c = re.sub(r'<[^>]+>', '', c).strip()
                processed.append(c)
            # filter out outputs that look like instructions or prompt-echo (model returned requirements instead of context)
            def looks_like_instruction(text: str) -> bool:
                txt = text.strip()
                instr_tokens = ['need', 'needs', 'required', 'require', 'Each sentence', 'Each sentence needs', 'every sentence', 'required', 'target length', 'strict requirement', 'only return', 'example', 'do not']
                lower = txt.lower()
                for t in instr_tokens:
                    if t.lower() in lower:
                        return True

                if len(re.findall(r'task|requirement|strict', txt)) >= 1:
                    return True
                return False

            filtered = [s for s in processed if not looks_like_instruction(s)]

            # if filtering removed everything, attempt a focused re-prompt asking only for JSON array
            if not filtered and attempt < max_attempts:
                repair_prompt = (
                    "Your previous answer repeated task instructions or meta-information, "
                    "but I need actual candidate sentences.\n"
                    "Please strictly return only a JSON array, with each element being a full candidate sentence, "
                    "for example: [\"He hesitated before entering the room.\", \"The sound of rain separated them into two worlds.\"]. "
                    "Do not return any explanations, bullet points, or meta-information.\n"
                    f"Theme: {title}\nContext: {context}\ncontext part: {part_name}\nMaximum array length: {requested_k}."
                )
                system_p_repair = (
                    "Directly return a JSON array, with each element being a complete context sentence (single sentence). "
                    "Do not include any explanations, steps, or brainstorming text. "
                    "Do not include meta phrases such as 'I need', 'the user wants', 'Let', 'Let's', or '<think>'."
                )
                raw = await self._call_hosted_model(repair_prompt, temperature=tem, max_tokens=model_max_tokens, system_prompt=system_p_repair)
                if getattr(self, 'debug', False):
                    print(f"\n[DEBUG] raw repair response for part={part_name}, attempt={attempt}:")
                    print(raw)
                # try to parse repaired response into candidates immediately
                try:
                    start = raw.find('[')
                    end = raw.rfind(']')
                    if start != -1 and end != -1 and end > start:
                        arr = json.loads(raw[start:end+1])
                        if isinstance(arr, list):
                            processed = [str(x).strip() for x in arr]
                            filtered = [s for s in processed if not looks_like_instruction(s)]
                except Exception:
                    filtered = []

            # set final_candidates from filtered when available
            if filtered:
                final_candidates = filtered[:requested_k]
                break

            final_candidates = processed[:requested_k]
            if final_candidates:
                break

        # If still empty, fallback to original generic parsing of the last raw
        if not final_candidates:
            try:
                start = raw.find('[')
                end = raw.rfind(']')
                if start != -1 and end != -1 and end > start:
                    arr = json.loads(raw[start:end+1])
                    if isinstance(arr, list):
                        final_candidates = [str(x).strip() for x in arr][:k]
            except Exception:
                final_candidates = []

        return final_candidates[:k]
    
    async def _evaluate_state(self, full_text: str, module_text: str, use_model: bool, part_name: Optional[str] = None) -> Tuple[float, float, float, float, float]:

            raw_key = f"{self.outline_text}|{full_text}|{module_text}|{part_name}"
            cache_key = hashlib.sha1(raw_key.encode("utf-8")).hexdigest()

            if cache_key in self.eval_cache:
                return self.eval_cache[cache_key]


            if part_name and part_name in self.part_hints:
                outline_hints = json.dumps(self.part_hints[part_name], indent=2, ensure_ascii=False)
            else:
                outline_hints = self.outline_text if self.outline_text else "No specific outline provided."

            prompt = EVALUATION_PROMPT.format(full_text=full_text, module_text=module_text, outline_hints=outline_hints)


            raw = await self._call_hosted_model(prompt, temperature=0.0, max_tokens=256, system_prompt="Return ONLY JSON.")
            

            obj = self._json_extract_object(raw)
            scaff = self._clamp01(float(obj.get("scaffolding_and_adherence", 0.0)))
            img = self._clamp01(float(obj.get("image", 0.0)))
            coh = self._clamp01(float(obj.get("coherence", 0.0)))
            hall = self._clamp01(float(obj.get("hallucination", 0.0)))


            reward = (self.w_scaff * scaff) + (self.w_img * img) + (self.w_coh * coh)
            reward = reward * (1.0 - hall)  
            reward = self._clamp01(reward)

            # Store in cache and return
            result = (scaff, img, coh, hall, reward)
            self.eval_cache[cache_key] = result
            return result

    async def _evaluate_rollout(self, s, action, max_steps, part_name): 
        simulated_state = s + [action]
        full_text = "\n".join(simulated_state)
        module_text = action
        _, _, _, _, reward = await self._evaluate_state(full_text, module_text, use_model=True, part_name=part_name)
        threshold = 0.5
        if reward < threshold:
            sim_state = await self._rollout(s, action, max_steps, part_name=part_name) 
            full_text2 = "\n".join(sim_state)
            module_text2 = action
            _, _, _, _, reward = await self._evaluate_state(full_text2, module_text2, use_model=True, part_name=part_name)
        return reward

    async def _rollout(self, state: List[str], action: str, max_steps: int, part_name: str) -> List[str]:
        simulated_state = state + [action]
        for _ in range(2):  # Generate two steps
            candidates = await self._generate_candidates("","", part_name, simulated_state, 1)
            if not candidates:
                break
            simulated_state.append(candidates[0])
        return simulated_state


    async def generate_serial(self, title: str, theme: str = "", parts: Optional[List[str]] = None,
                              num_simulations: int = 10, num_candidates: int = 4) -> Dict[str, str]:
        
        parts = parts or ["Anchor", "Scene Setting", "Characters & Interaction", "Conflict & Challenge", "Open Task"]
        generated = {}
        state = [f"Title: {title}", f"Theme: {theme}"]

        for part in parts:
            tpl = PART_TEMPLATES.get(part, {})
            target_sentences = int(tpl.get('target_sentences', 1)) 

            print(f" Starting MCTS path planning for part [{part}] (depth: {target_sentences})...")


            async def gen_actions(s, k):
                return await self._generate_candidates(title, theme, part, s, k)

            dynamic_sims = num_simulations + (target_sentences * 30)
            tree = AsyncMCTS(
                generate_actions_fn=gen_actions,
                rollout_fn=lambda s, a, ms: self._evaluate_rollout(s, a, ms, part), 
    
                evaluate_state_fn=lambda ft, mt, use_model, pn=part: self._evaluate_state(ft, mt, use_model, pn),
                num_simulations=dynamic_sims,
                num_expansion_actions=num_candidates
            )
            # One search, directly get the entire part's sentence list
            best_part_sentences = await tree.search(list(state), target_depth=target_sentences)

            state.extend(best_part_sentences)
            generated[part] = "\n".join(best_part_sentences).strip()
            print(f" Part [{part}] generation completed, {len(best_part_sentences)} sentences.")

        return generated


async def generate_mcts_context(title: str,
                              theme: str,
                              output_file: str = "outputs/initial_contexts.json",
                              num_contexts: int = 2,
                              outline_text: str = None,
                              few_shot_example: str = None):   

    start_time = time.time()
    model_kwargs = DEFAULT_HOSTED_MODEL_KWARGS
    generator = MCTSContext(
        model_kwargs=model_kwargs,
        outline_text=outline_text,
        few_shot_example=few_shot_example  
    )
    parts = ["Anchor", "Scene Setting", "Characters & Interaction", "Conflict & Challenge", "Open Task"]

    all_contexts = []

    for i in range(num_contexts):
        print(f" Generating context {i + 1}/{num_contexts}...")

        full_title = title

        out = await generator.generate_serial(
            title=full_title,
            theme=theme,
            parts=parts,
            num_simulations=10,
            num_candidates=3
        )

        context = {
            "Anchor": out.get("Anchor", ""),
            "Scene Setting": out.get("Scene Setting", ""),
            "Characters & Interaction": out.get("Characters & Interaction", ""),
            "Conflict & Challenge": out.get("Conflict & Challenge", ""),
            "Open Task": out.get("Open Task", "")
        }

        all_contexts.append(context)

    # Save as JSON file
    Path("outputs").mkdir(exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({"contexts": all_contexts}, f, ensure_ascii=False, indent=2)

    elapsed = (time.time() - start_time) / 60
    print(f" Generated {num_contexts} contexts, saved to {output_file}, took {elapsed:.2f} minutes")

    return output_file


async def generate_MCTSContext(number: int, num_contexts: int):
    input_path = Path("modules/outputs/outline") / f"generated_context_{number}.json"
    if not input_path.exists():
        raise FileNotFoundError(f"File not found: {input_path}")
    with open(input_path, "r", encoding="utf-8") as f:
        item = json.load(f)
    title = item.get("title", f"untitled_{number}")
    theme = item.get("theme", title)
    outline = item.get("outline", "") or ""
    current_dir = Path(__file__).parent
    save_dir = current_dir / "outputs" / "mcts_htp"
    save_dir.mkdir(parents=True, exist_ok=True)
    out_path = save_dir / f"mcts_{number}.json"

    print(f"\n Running mcts (referencing HTP outline): {title}")
    t0 = time.time()
    out_file = await generate_mcts_context(
        title=title,
        theme=theme,
        output_file=str(out_path),
        num_contexts=num_contexts,
        outline_text=outline
    )
    print(f" Context #{number} generation completed, elapsed time {time.time() - t0:.2f}s -> {out_file}")
    return out_file
