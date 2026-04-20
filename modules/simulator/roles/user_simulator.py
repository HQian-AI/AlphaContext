from typing import List
import logging
import litellm
import os.path as osp

from prompts import COLLABLLM_TERMINATION_SIGNAL
from modules.simulator.src.utils.template import parse_messages
from modules.simulator.src.utils.extract_json_reliable import extract_json
logger = logging.getLogger(__name__)


class UserSimulator(object):
    def __init__(self, task_desc='', single_turn_prompt='', num_retries=10, prompt_variant='quiet', prompt_path=None, **llm_kwargs):
        super().__init__()
        self.task_desc = task_desc
        self.single_turn_prompt = single_turn_prompt
        self.num_retries = num_retries

        self.llm_kwargs = {"temperature": 1.0, "max_tokens": 2048, **llm_kwargs}
        
        assert 'model' in self.llm_kwargs, "Model name must be provided in llm_kwargs"

        # prompt selection: either explicit path or variant name
        self.prompt_variant = prompt_variant
        self._prompt_template = None
        if prompt_path:
            try:
                with open(prompt_path, 'r', encoding='utf-8') as f:
                    self._prompt_template = f.read()
            except Exception:
                logger.warning(f"[UserSimulator] Failed to load prompt from path: {prompt_path}")
        if self._prompt_template is None:
            self._prompt_template = self._load_prompt_variant(prompt_variant)

    def _load_prompt_variant(self, variant: str) -> str:

        mapping = {
            'quiet': 'user_simulator_quiet.txt',
            'talkative': 'user_simulator_talkative.txt',
            'default': 'user_simulator.txt',
        }
        fname = mapping.get(variant, 'user_simulator_quiet.txt')
        # Resolve against the project-root prompts/ directory.
        base_dir = osp.abspath(osp.join(osp.dirname(__file__), '..', '..', '..', 'prompts'))
        path = osp.join(base_dir, fname)
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception:
            logger.warning(f"[UserSimulator] Could not load prompt file {path}, falling back to quiet builtin.")
            # fallback: try to import default from prompts package if available
            try:
                from prompts import USER_SIMULATOR_PROMPT
                return USER_SIMULATOR_PROMPT
            except Exception:
                return "You are a user simulator. {task_desc} {single_turn_prompt} {chat_history} {terminal_signal}"

    def __call__(self, messages: List[dict]):
        # format prompt using the selected template
        prompt = self._prompt_template.format(
            task_desc=self.task_desc,
            single_turn_prompt=self.single_turn_prompt,
            chat_history=parse_messages(messages, strip_sys_prompt=True),
            terminal_signal=COLLABLLM_TERMINATION_SIGNAL,
        )
        messages = [{"role": "user", "content": prompt}]

        response = ""
        for _ in range(self.num_retries):
            print(self.llm_kwargs)
            full_response = litellm.completion(
                **{k: v for k, v in self.llm_kwargs.items() if v is not None},
                messages=messages,
                num_retries=self.num_retries,
            ).choices[0].message.content
            try:
                if isinstance(full_response, str):
                    full_response = extract_json(full_response)
            except Exception as e:
                logger.error(f"[UserSimulator] Error extracting JSON: {e}")
                continue

            if isinstance(full_response, dict):
                keys = full_response.keys()
                if {'current_answer', 'thought', 'response'}.issubset(keys):
                    response = full_response.pop('response')
                    break
                else:
                    logger.error(f"[UserSimulator] Keys {keys} do not match expected keys. Retrying...")
                    continue
        return response.strip()
