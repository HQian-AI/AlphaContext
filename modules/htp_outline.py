from langchain.prompts import PromptTemplate

from prompts.hypertree_prompts import (
    select_prompt,
    seed_select_prompt,
    optional_pick_prompt,
    planner_agent_prompt,
)
import tiktoken
from difflib import get_close_matches
import random
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
try:
    from typing import LiteralString
except ImportError:
    from typing_extensions import LiteralString
import requests, json
from model_config import MODEL_NAME_deepseek, CHAT_EP_deepseek, get_deepseek_headers
K = 5

def llm_chat(
    user_prompt: str,
    *,
    system_prompt: str = None,
    temperature: float = 0.0,
    max_tokens: int = 2048,
) -> str:
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})

    payload = {
        "model": MODEL_NAME_deepseek,
        "messages": messages,
        "temperature": float(temperature),
        "max_tokens": int(max_tokens),
        "stream": False,
    }
    headers = get_deepseek_headers()
    try:
        resp = requests.post(CHAT_EP_deepseek, json=payload, headers=headers, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, dict) and data.get("choices"):
            c0 = data["choices"][0]
            if isinstance(c0, dict) and isinstance(c0.get("message"), dict):
                return c0["message"].get("content", "") or ""
    except requests.exceptions.RequestException as e:
        print(" API error:", e)
    return ""


class Planner:
    def __init__(self,
                 agent_prompt: PromptTemplate = planner_agent_prompt,
                 model_name: str = MODEL_NAME_deepseek,
                 ) -> None:

        self.agent_prompt = agent_prompt
        self.scratchpad: str = ''
        self.model_name = model_name


        try:
            self.enc = tiktoken.get_encoding("cl100k_base")
        except Exception:
            self.enc = None
        self.max_token_length = 30000

        print(f"PlannerAgent (backend={self.model_name}) loaded.")

    def _chat(self, user_prompt: str, system_prompt: str = None,
              temperature: float = 0.2, max_tokens: int = 2048) -> str:

        return llm_chat(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        ) or ""

    def run(self, title, theme, log_file=None) -> str:
        prompt = self._build_agent_prompt(title, theme)


        if self.enc is not None:
            try:
                if len(self.enc.encode(prompt)) > 12000:
                    return 'Max Token Length Exceeded.'
            except Exception:
                pass

        if log_file:

            log_file.write('\n---------------Planner\n' + prompt)


        reply = self._chat(prompt, temperature=0.2, max_tokens=2048)
        return reply

    def _build_agent_prompt(self, title, theme) -> str:
        return self.agent_prompt.format(
            title=title,
            theme=theme
        )


class HyperTree:
    def __init__(self, value: str, rules: Optional[Dict[str, Any]] = None):
        self.value: str = value
        self.rules: Dict[str, Any] = rules or {}
        self.edges: List[List[str]] = []
        self.edge_children: List[List["HyperTree"]] = []


    def is_expandable(self) -> bool:
        name = self.value.strip()
        if not self.rules:
            return False
        return name in self.rules

    def is_terminal(self) -> bool:
        name = self.value.strip()
        return not (self.rules and name in self.rules)


    def attach_edge(self, child_set: List[str]) -> None:
        if not child_set:
            return
        self.edges.append(child_set)
        children_nodes = [HyperTree(v, rules=self.rules) for v in child_set]
        self.edge_children.append(children_nodes)

    def show_full(self, depth: int = 0, show_edges: bool = False) -> str:

        prefix = "<Tab>" * depth
        s = f"{prefix}{self.value}\n"

        if not self.edge_children:
            return s

        if show_edges:
            for ei, children in enumerate(self.edge_children):
                s += "<Tab>" * (depth + 1) + f"(edge {ei})\n"
                for ch in children:
                    s += ch.show_full(depth + 2, show_edges=show_edges)
        else:

            seen_ids = set()
            for children in self.edge_children:
                for ch in children:
                    cid = id(ch)
                    if cid in seen_ids:
                        continue
                    seen_ids.add(cid)
                    s += ch.show_full(depth + 1, show_edges=show_edges)

        return s

    def show_instantiated(self, choice_map: Dict[str, int], depth: int = 0) -> str:
        prefix = "<Tab>" * depth
        s = f"{prefix}{self.value}\n"

        key = HyperTree._node_key(self)
        if key in choice_map:
            ei = choice_map[key]
            if 0 <= ei < len(self.edge_children):
                for ch in self.edge_children[ei]:
                    s += ch.show_instantiated(choice_map, depth + 1)
        return s

    @staticmethod
    def _node_key(node: "HyperTree") -> str:
        return node.value  


class HTPlanner:
    def __init__(self):

        self.select_prompt = select_prompt

        self.CHALLENGE_SEED_RANGES = {
            '[Challenge Seeds 1]': (2, 3),
            '[Challenge Seeds 2]': (2, 3),
            '[Challenge Seeds 3]': (3, 4),
            '[Challenge Seeds 4]': (4, 5),
        }
        self.OPTIONAL_HYPER_NODES = {
            '[Future Horizon]',
            '[Scenario Frame]',
            '[Scale]',
            '[Constraint Hints]',
            '[Interaction Goal]',
            '[Dispute Focus]',
            '[Creativity Triggers]',
        }


        self.OPTIONAL_PICK_RANGES: Dict[str, Tuple[int, int]] = {
            '[Future Horizon]': (1, 1),
            '[Scenario Frame]': (1, 1),
            '[Scale]': (1, 1),
            '[Interaction Goal]': (1, 1),
            '[Dispute Focus]': (1, 1),
            '[Constraint Hints]': (2, 3),
            '[Creativity Triggers]': (1, 3),
        }


        self.MAP_M = 8
        self.FILTER_W = 4

        self.query = None
        self.rules: Dict[str, Any] = {}
        self.root: Optional[HyperTree] = None


        self.CHALLENGE_SEED_POOL: List[str] = []
        self.selected_domain_tags: List[str] = []
        self.selected_challenge_seeds: List[str] = []


    def _chat(self, user_prompt: str, system_prompt: str = None,
              temperature: float = 0.2, max_tokens: int = 2048) -> str:

        return llm_chat(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        ) or ""

    @staticmethod
    def _extract_json_array(text: str):
        if not text:
            return []

        try:
            obj = json.loads(text)
            return obj if isinstance(obj, list) else []
        except Exception:
            pass

        m = re.search(r'\[.*\]', text, flags=re.S)
        if not m:
            return []
        try:
            obj = json.loads(m.group(0))
            return obj if isinstance(obj, list) else []
        except Exception:
            return []

    @staticmethod
    def _parse_int(reply: str, n: int) -> int:
        if not reply:
            return 0
        m = re.search(r'\d+', reply)
        if not m:
            return 0
        idx = int(m.group(0))
        if 0 <= idx < n:
            return idx
        if 1 <= idx <= n:
            return idx - 1
        return max(0, min(n - 1, idx))

    def _build_seed_select_prompt(self, min_k: int, max_k: int, node_name: str) -> str:
        full_tree_str = self.root.show_full(show_edges=False)
        if len(full_tree_str) > 6000:
            full_tree_str = full_tree_str[:6000] + "\n...<TRUNCATED>...\n"

        return seed_select_prompt.format(
            title=self.query["title"],
            theme=self.query["theme"],
            full_tree=full_tree_str,
            node_name=node_name,
            candidate_seeds=", ".join(self.CHALLENGE_SEED_POOL),
            min_k=min_k,
            max_k=max_k,
        )

    def _select_challenge_seeds_via_llm(self, min_k: int, max_k: int, node_name: str) -> List[str]:
        raw = self._chat(self._build_seed_select_prompt(min_k, max_k, node_name), temperature=0.2)
        arr = self._extract_json_array(raw)

        def _norm_tag(s: str) -> str:
            s = s.strip()
            s = s.replace("–", "-").replace("—", "-")
            s = s.replace("&", "and")
            s = s.replace("/", " / ")
            s = re.sub(r"\s+", " ", s)
            return s.lower()


        pool_map = {_norm_tag(c): c for c in self.CHALLENGE_SEED_POOL}

        picked = []
        for s in arr:
            if not isinstance(s, str):
                continue
            key = _norm_tag(s)
            if key in pool_map:
                picked.append(pool_map[key])
            else:

                m = get_close_matches(key, pool_map.keys(), n=1, cutoff=0.88)
                if m:
                    picked.append(pool_map[m[0]])


        seen, out = set(), []
        for x in picked:
            if x not in seen:
                seen.add(x)
                out.append(x)
        return out

    def _select_problem_topic_phrase(self) -> str:
        title = (self.query.get("title") or "").strip()
        theme = (self.query.get("theme") or "").strip()
        prompt = f"""
                You are a creativity assessment and scenario design expert.
                
                Title: {title}
                Theme: {theme}
                
                Task:
                - Extract ONE short English noun phrase that best describes the core problem topic.
                - Maximum 6–8 words.
                Return ONLY the noun phrase.
                """
        raw = self._chat(prompt, temperature=0.3, max_tokens=64).strip()

        raw = re.sub(r'^[\s"\']+|[\s"\']+$', '', raw)
        if not raw:
            raw = theme or title or "Core problem topic"
        if len(raw) > 80:
            raw = raw[:80].rstrip()
        return raw

    def _select_challenge_seeds_for_node(self, node_value: str) -> List[str]:
        min_k, max_k = self.CHALLENGE_SEED_RANGES.get(node_value, (2, 4))
        min_k = max(0, min_k)
        max_k = max(min_k, max_k)


        desired_k = random.randint(min_k, max_k) if max_k > 0 else 0
        desired_k = min(desired_k, len(self.CHALLENGE_SEED_POOL))

        arr = self._select_challenge_seeds_via_llm(min_k, max_k, node_value)


        if not arr:
            return random.sample(self.CHALLENGE_SEED_POOL, k=desired_k) if desired_k > 0 else []


        if len(arr) < desired_k:
            remaining = [s for s in self.CHALLENGE_SEED_POOL if s not in arr]
            random.shuffle(remaining)
            arr = arr + remaining[: (desired_k - len(arr))]

        return arr[:desired_k]

    @staticmethod
    def _parse_bracket_rule(rule: Any) -> List[str]:
        if not rule:
            return []
        if isinstance(rule, str):
            core = rule.strip()
            if not core:
                return []

            items = [x for x in core.strip("[]").split("][") if x]
            return [f"[{x}]" for x in items]
        if isinstance(rule, list):
            out = []
            for it in rule:
                if not it:
                    continue
                it = str(it)
                out.append(it if (it.startswith("[") and it.endswith("]")) else f"[{it}]")
            return out
        return []


    def _pick_from_pool_via_llm(self, node_name: str, base_children: List[str], min_k: int, max_k: int) -> List[str]:
        min_k = max(0, min_k)
        max_k = max(min_k, max_k)
        max_k = min(max_k, len(base_children))
        min_k = min(min_k, max_k)
        fallback_k = random.randint(min_k, max_k) if max_k > 0 else 0

        full_tree_str = self.root.show_full(show_edges=False)
        if len(full_tree_str) > 6000:
            full_tree_str = full_tree_str[:6000] + "\n...<TRUNCATED>...\n"

        def _range_text(min_k: int, max_k: int) -> str:
            if min_k == max_k:
                if min_k == 1:
                    return "Select EXACTLY 1 item (the single BEST choice)."
                return f"Select EXACTLY {min_k} items."
            return f"Select between {min_k} and {max_k} items (inclusive)."

        range_text = _range_text(min_k, max_k)
        order_rule = (
            "IMPORTANT: Return items sorted from MOST relevant to LEAST relevant.\n"
            if not (min_k == max_k == 1)
            else "IMPORTANT: Choose the single most relevant item.\n"
        )
        candidates_str = "\n".join(base_children)

        prompt = optional_pick_prompt.format(
            title=self.query["title"],
            theme=self.query["theme"],
            full_tree=full_tree_str,
            node_name=node_name,
            candidates=candidates_str,
            range_text=range_text,
            order_rule=order_rule,
        )
        raw = self._chat(prompt, temperature=0.3, max_tokens=256)
        arr = self._extract_json_array(raw)

        def _norm(s: str) -> str:
            s = s.strip()
            s = s.replace("–", "-").replace("—", "-")
            s = re.sub(r"\s+", " ", s)
            return s.lower()


        cand_map = {_norm(c): c for c in base_children}

        picked = []
        for s in arr:
            if not isinstance(s, str):
                continue
            key = _norm(s)
            if key in cand_map:
                picked.append(cand_map[key])
            else:

                m = get_close_matches(key, cand_map.keys(), n=1, cutoff=0.88)
                if m:
                    picked.append(cand_map[m[0]])


        seen, out = set(), []
        for x in picked:
            if x not in seen:
                seen.add(x)
                out.append(x)

        arr = out

        if len(out) == 0 and fallback_k > 0:
            return random.sample(base_children, k=fallback_k)

        if len(out) > max_k:
            out = out[:max_k]

        if len(out) < min_k:
            remain = [c for c in base_children if c not in seen]
            random.shuffle(remain)
            out.extend(remain[: (min_k - len(out))])


        if len(out) < fallback_k:
            remain = [c for c in base_children if c not in out]
            random.shuffle(remain)
            out.extend(remain[: (fallback_k - len(out))])

        return out


    def _map_sample(self, m: int) -> List[Dict[str, int]]:
        out = []
        for _ in range(max(1, m)):
            choice_map: Dict[str, int] = {}

            def dfs(node: HyperTree, path: str):

                if node.edges:
                    ei = random.randrange(len(node.edges))
                    choice_map[path] = ei
                    for ci, ch in enumerate(node.edge_children[ei]):
                        dfs(ch, path + f"/{ci}:{ch.value}")


            dfs(self.root, f"0:{self.root.value}")
            out.append(choice_map)
        return out


    def _filter_chains(self, chains: List[Dict[str, int]], W: int) -> List[Dict[str, int]]:
        if not chains:
            return []
        if len(chains) <= W:
            return chains


        rendered = []
        for i, cm in enumerate(chains):
            rendered.append(f"=== Chain {i} ===\n{self._render_choice_map(cm)}")

        prompt = (
            "You are a planner selecting which partial outlines are most promising to expand next.\n\n"
            f"Title: {self.query['title']}\n"
            f"Theme: {self.query['theme']}\n\n"
            f"Pick the best {W} chains (by index) considering coherence and potential.\n"
            "Return ONLY a JSON array of integers.\n\n"
            + "\n\n".join(rendered)
        )
        raw = self._chat(prompt, temperature=0.2, max_tokens=512)
        idxs = self._extract_json_array(raw)
        idxs = [x for x in idxs if isinstance(x, int) and 0 <= x < len(chains)]

        seen, pick = set(), []
        for x in idxs:
            if x not in seen:
                seen.add(x)
                pick.append(x)
        if len(pick) < W:
            remain = [i for i in range(len(chains)) if i not in seen]
            random.shuffle(remain)
            pick.extend(remain[: (W - len(pick))])
        pick = pick[:W]
        return [chains[i] for i in pick]


    def _collect_divisible_leaves(self, choice_map: Dict[str, int]) -> List[Tuple[str, HyperTree]]:

        leaves: List[Tuple[str, HyperTree]] = []

        def dfs(node: HyperTree, path: str):
            if node.edges:
                ei = choice_map.get(path, None)
                if ei is None or not (0 <= ei < len(node.edge_children)):
                    return
                for ci, ch in enumerate(node.edge_children[ei]):
                    dfs(ch, path + f"/{ci}:{ch.value}")
            else:
                if node.is_expandable():
                    leaves.append((path, node))

        dfs(self.root, f"0:{self.root.value}")
        return leaves

    def _select_leaf_index_via_llm(self, current_tree_str: str, leaves: List[Tuple[str, HyperTree]]) -> int:
        leaves_dict = {i: node.value for i, (_, node) in enumerate(leaves)}
        prompt = self.select_prompt.format(
            title=self.query["title"],
            theme=self.query["theme"],
            current_tree=current_tree_str.rstrip("\n"),
            leaves=leaves_dict,
        )
        reply = self._chat(prompt, temperature=0.1, max_tokens=32)
        return self._parse_int(reply, len(leaves))

    def _expand_and_attach_once(self, node: HyperTree) -> None:
        name = node.value


        if name == "[Problem Slot]":
            phrase = self._select_problem_topic_phrase()
            if phrase:
                node.attach_edge([phrase])
            return


        if name.startswith("[Challenge Seeds"):
            seeds = self._select_challenge_seeds_for_node(name)
            if seeds:
                parts = [f"[{s}]" for s in seeds]
                node.attach_edge(parts)
            return


        if name not in self.rules:
            return

        base_children = self._parse_bracket_rule(self.rules.get(name))
        if not base_children:
            return


        if name in self.OPTIONAL_HYPER_NODES:
            min_k, max_k = self.OPTIONAL_PICK_RANGES.get(name, (1, 1))
            picked = self._pick_from_pool_via_llm(name, base_children, min_k, max_k)
            if picked:
                node.attach_edge(picked)
            return

        node.attach_edge(base_children)


    def _render_choice_map(self, choice_map: Dict[str, int]) -> str:
        def dfs(node: HyperTree, path: str, depth: int) -> str:
            s = "<Tab>" * depth + node.value + "\n"
            if node.edges:
                ei = choice_map.get(path, None)
                if ei is not None and 0 <= ei < len(node.edge_children):
                    for ci, ch in enumerate(node.edge_children[ei]):
                        s += dfs(ch, path + f"/{ci}:{ch.value}", depth + 1)
            return s
        return dfs(self.root, f"0:{self.root.value}", 0)


    def run(self, query: Dict[str, str]) -> str:

        self.query = query


        self.CHALLENGE_SEED_POOL = [
            "Arts & Aesthetics", "Basic Needs", "Business & Commerce", "Communication",
            "Culture & Religion", "Defense", "Economics", "Education", "Environment",
            "Ethics & Morality", "Government & Politics", "Law & Justice",
            "Physical Health", "Psychological Health", "Recreation", "Science",
            "Social Relationships", "Technology", "Transportation"
        ]



        horizon_options = [
            "NearFuture (5–15 years)",
            "MidFuture (15–40 years)",
            "FarFuture (40+ years)",
            "Speculative"
        ]
        future_horizon_rule = "[" + "][".join(horizon_options) + "]"

        scenario_frame_options = [
            "Everyday Life",
            "City Infrastructure",
            "Virtual / Mixed Reality",
            "Planetary / Space",
            "Organizational / Workplace",
            "Education Setting",
        ]
        scenario_frame_rule = "[" + "][".join(scenario_frame_options) + "]"


        constraint_hint_options = [
            "Policy", "Budget", "Time Limit", "Safety",
            "Resource Scarcity", "Data / Privacy",
        ]
        constraint_rule = "[" + "][".join(constraint_hint_options) + "]"

        creativity_trigger_options = [
            "Uncertainty Cue", "Contradiction Cue",
            "Resource Constraint Cue",
        ]
        creativity_triggers_rule = "[" + "][".join(creativity_trigger_options) + "]"

        interaction_goal_options = [
            "Co-creation Workshop", "Negotiation Meeting", "Emergency Response",
            "Design Review", "Public Hearing", "Future Planning Session",
        ]
        interaction_goal_rule = "[" + "][".join(interaction_goal_options) + "]"

        dispute_focus_options = [
            "Value Conflict", "Resource Conflict", "Trust Conflict",
            "Role Responsibility Conflict", "Vision Conflict",
        ]
        dispute_focus_rule = "[" + "][".join(dispute_focus_options) + "]"
        Scale_options = [
            "Community", "National", "International",
            "Space",
        ]
        Scale_rule = "[" + "][".join(Scale_options) + "]"

        self.rules = {
            "[Plan]": "[Anchor][Scene Setting][Characters & Interaction][Conflict & Challenge][Open Task]",

            "[Anchor]": "[Future Horizon][Place][Scale][Challenge Seeds 1]",
            "[Future Horizon]": future_horizon_rule,
            "[Place]": "[City Or Region][Specific Facility]",
            "[Scale]": Scale_rule,

            "[Scene Setting]": "[Scenario Frame][Constraint Hints][Challenge Seeds 2]",
            "[Scenario Frame]": scenario_frame_rule,
            "[Constraint Hints]": constraint_rule,

            "[Characters & Interaction]": "[Interaction Goal][Dispute Focus][Problem Slot][Challenge Seeds 3]",
            "[Interaction Goal]": interaction_goal_rule,
            "[Dispute Focus]": dispute_focus_rule,
            "[Problem Slot]": "[Topic Phrase]",

            "[Conflict & Challenge]": "[Challenge Seeds 4][Creativity Triggers]",
            "[Creativity Triggers]": creativity_triggers_rule,

            "[Open Task]": "[Challenge Identification][Solution Exploration]",
            "[Challenge Identification]": "[Prompt student to identify multiple challenges in the scenario]",
            "[Solution Exploration]": "[Prompt student to think of possible response strategies]",


            "[Challenge Seeds 1]": "",
            "[Challenge Seeds 2]": "",
            "[Challenge Seeds 3]": "",
            "[Challenge Seeds 4]": "",
        }

        self.root = HyperTree("[Plan]", rules=self.rules)
        self._expand_and_attach_once(self.root)

        for _ in range(max(1, K)):
            chains = self._map_sample(self.MAP_M)
            chains = self._filter_chains(chains, self.FILTER_W)

            for cm in chains:
                divisible_leaves = self._collect_divisible_leaves(cm)
                if not divisible_leaves:
                    continue

                current_tree_str = self._render_choice_map(cm)
                pick_i = self._select_leaf_index_via_llm(current_tree_str, divisible_leaves)
                _, gstar = divisible_leaves[pick_i]


                self._expand_and_attach_once(gstar)


        final_chains = self._map_sample(max(1, self.MAP_M))
        best = self._filter_chains(final_chains, 1)[0]
        self.best_choice_map = best


        return self.root.show_full(show_edges=False).rstrip("\n")


def _extract_json_array(text: str):

    try:
        m = re.search(r'\[.*?\]', text, flags=re.S)
        if m:
            return json.loads(m.group(0))
        return json.loads(text)
    except Exception:
        return []


def _extract_json_hyperedges(text: str):

    try:
        m = re.search(r'\[.*\]', text, flags=re.S)
        if not m:
            return []
        data = json.loads(m.group(0))
        if not isinstance(data, list) or not data:
            return []

        if isinstance(data[0], str):
            return [[s] for s in data if isinstance(s, str) and s.strip()]

        edges = []
        for edge in data:
            if isinstance(edge, list):
                cleaned = [s for s in edge if isinstance(s, str) and s.strip()]
                if cleaned:
                    edges.append(cleaned)
        return edges
    except Exception:
        return []


def generate_outline(query_data, number):

    planner = HTPlanner()
    try:
        outline = planner.run(query_data)
        result = {
            "idx": number,
            "title": query_data["title"],
            "theme": query_data["theme"],
            "outline": outline
        }
    except Exception as e:
        print(f"Error on sample #{number}: {e}")
        result = {
            "idx": number,
            "title": query_data.get("title", "Unknown"),
            "theme": query_data.get("theme", "Unknown"),
            "error": str(e)
        }

    output_path = Path(__file__).parent / "outputs" / "outline" / f"generated_context_{number}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4, ensure_ascii=False)

    return result
