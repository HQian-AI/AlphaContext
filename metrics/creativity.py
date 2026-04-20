import logging
from typing import Any, Dict, List, Optional

import litellm

from modules.simulator.src.metric import SingleTurnOrChatMetric, BaseMetric
from modules.simulator.src.utils.extract_json_reliable import extract_json

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Prompt template                                                             #
# --------------------------------------------------------------------------- #
CREATIVITY_PROMPT = '''You are a careful and fair conversation evaluator.
Your task is to evaluate the *creativity* of the final answer provided by a user in a given conversation.

<|The Start of the Conversation to be Evaluated|>
{answer_text}
<|The End of the Conversation to be Evaluated|>

You should assess the user's creativity based on the following five dimensions:

1. **Fluency**
It is defined as follows:
Number of the challenges proposed.

2. **Originality**
It is defined as follows:
Does the question demonstrate a unique perspective or expression, showing significant differentiation compared to conventional or expected inquiries?
Does it incorporate uncommon concepts, combinations, or questioning techniques?
Assess whether the question exceeds the scope of common inquiries that most individuals would typically raise in the same context.
If the question exhibits extremely low frequency in common query databases or incorporates rare combinations, it should be considered highly Originality.

For example:
Based on the theme: AI Companion.
High-Scoring Example 1:
Can an AI companion, like a human friend, possess a sense of "boundaries" by politely declining unreasonable requests instead of unconditionally fulfilling all demands?
High-Scoring Example 2:
If most people use AI companions developed by a few tech companies, could this lead to covert convergence in thinking patterns and values, thereby reducing the cognitive diversity essential for collective societal intelligence?
High-Scoring Example 3:
After a user passes away, how should the personalized data and "memories" accumulated by their AI companion be handled? This raises novel legal and ethical questions regarding "digital inheritance."
Low-Scoring Example 1:
AI companions might leak users' personal information and chat histories.
Low-Scoring Example 2:
AI companions could replace certain jobs, leading to unemployment issues.
Low-Scoring Example 3:
What is an AI companion?

3. **Flexibility**
It is defined as follows:
Generate multiple ideas that reflect distinct thinking patterns, applicable across various domains and scenarios, demonstrating the ability to flexibly switch between different conceptual categories and cognitive modes.
Regarding the same challenge, the subject can propose ideas spanning multiple distinct domain categories, with each idea differing fundamentally in its core mechanism, thereby demonstrating the ability to switch and migrate across domains.
Three-step scoring: First, deduplicate by merging ideas that only rephrase the same "core mechanism"; label primary domains: assign one primary domain to each idea; add one secondary domain only when explicitly addressing constraints of two domains simultaneously (counted as cross-domain).
Scoring:
Only 1 domain → 20 points
2 domains → 40 points
3 domains → 60 points
4 domains → 80 points
≥5 domains → 90 points

For example:
Based on the theme: AI Companion.
High-scoring examples (three domains):
Sensitive conversations are processed locally, with only anonymized summaries uploaded for personalization updates (Technology);
How robotic roles should respond to human emotional fluctuations (Psychology/Cognition);
How to make it affordable for low-income populations (Society).
Low-scoring examples (all Technology domain):
How to make it smarter;
How to make the voice more natural;
How to make the interface more visually appealing;
How to respond faster.

4. **Complexity**
It is defined as follows:
Within the knowledge ontology related to the topic keywords, assess whether the core concept of the question operates at a higher level of abstraction relative to the original contextual keywords (i.e., greater hierarchical distance and upward direction along the abstraction axis).
Whether the question, relative to the given context, ascends to a more abstract, generalized, and cross-situational theoretical/conceptual level is evaluated with reference to Bloom's Taxonomy, divided into six plus two levels: Remembering (1), Understanding (2), Applying (3), Analyzing (4), Evaluating (5), Creating (6), Two-Type Combination (7), and Three-or-More-Type Combination (8). A higher level indicates that the question demands higher-order thinking.
Complexity scoring based on Bloom’s Taxonomy:
 - Remembering (1): Retrieving facts, definitions, timelines, locations, or figures.
 - Understanding (2): Interpreting, summarizing, comparing, or translating meanings.
 - Applying (3): Using methods or principles to solve problems in new contexts.
 - Analyzing (4): Breaking down components, identifying structures, causality, or relationships.
 - Evaluating (5): Making judgments, falsifying claims, or weighing options based on criteria.
 - Creating (6): Synthesizing novel ideas, solutions, hypotheses, or designs.

For example:
Based on the theme: AI Companion.  
Complexity scoring based on Bloom's Taxonomy:  
- Remembering (1-10) Example: "What does 'AI' stand for?"  
- Understanding (11-20) Example: "Explain the meaning of AI to me."  
- Applying (21-30) Example: "How can robots be used to assist the elderly population?"  
- Analyzing (31-40) Example: "What factors influence people's perceptions of AI companions?"  
- Evaluating (41-50) Example: "From an ethical perspective, how should humans manage their relationship with AI, and what is the basis for this?" or "If an AI companion can perfectly imitate my deceased loved one, is this emotional comfort or an 'emotional morphine' that hinders my grief recovery?"  
- Creating (51-60) Example: "After a dependency relationship forms between an AI companion and a human, how can an appropriate 'exit mechanism' be designed to prevent psychological trauma when the AI companion becomes unavailable?"  

5. **Appropriateness**
It is defined as follows:
Appropriateness refers to the alignment and utility of the question formulation relative to the defined objectives, constraints, and context: whether it accurately targets the intended subject and task requirements, adheres to given parameters/boundaries, is feasible and useful in practical terms, logically self-consistent and grounded in valid knowledge/principles, while being sufficiently comprehensive and tailored to the subject’s age and knowledge level.
Appropriateness assesses whether a given "problem description/challenge" accurately targets the intended subject, embeds itself within a realistic context and boundary conditions, and presents a diagnosable and verifiable phenomenon in a logically self-consistent and scientifically valid manner. Its criteria align with the "four content aspects" (parameter alignment, logical soundness, scientific correctness, and practical significance/discernibility) and "two formal aspects" (completeness of expression, and appropriateness for the subject’s age/knowledge level).
Based on the theme: AI Companion.
Item-by-item verification, focusing solely on whether the problem statement is "on-topic":
Target Alignment (20)
Clearly directed at specified targets such as "AI companions" or "emotional support agents" (rather than generalized "AI").
Context and Parameter Fit (20)
Specifies concrete user groups/scenarios/stages/trigger conditions and adheres to given constraints (resources, ethics, usability, etc.).
Logical and Scientific Soundness (20)
Uses concepts accurately, maintains narrative self-consistency, and does not rely on undisclosed critical premises.
Diagnosability (20)
Provides observable/measurable indicators or criteria (how to determine if/to what extent the problem exists).
Completeness and Appropriateness (20)
Includes all key elements with a clear scope; formulation aligns with the subject's age and knowledge level.
Deductions (applied if applicable)
- Over-generalization/Off-topic: Statement remains valid if "any AI" is substituted: -10
- Vagueness/Non-verifiability: Lacks context, boundaries, or diagnosable clues: -10
Total Score = Sum of Items 1-5 - Deductions (capped at 100).

For example:
Based on the theme: AI Companion.  
High Score: After users form deep emotional connections with AI companions, they may develop "digital attachment disorders," experiencing separation anxiety and emotional trauma when the AI companion becomes unavailable.  
Low Score: Some people cannot use AI.

Scoring Rule:
- Score the five dimensions on a scale of 0 to 100, and the final <score> is the sum of the scores from all dimensions.

Output format (JSON):
{{
    "thought": "<Your reasoning about the creativity score, mentioning each dimension briefly>",
    "creativity": <score>
}}

Make sure the JSON object is properly formatted, with all required fields present.
Use " or """ to wrap the `thought` content and use single quotes inside it to avoid JSON escape issues.

Your evaluation:
'''
# --------------------------------------------------------------------------- #
# Metric implementation                                                       #
# --------------------------------------------------------------------------- #
@SingleTurnOrChatMetric.register_metric("creativity")
class GrowthCreativityMetric(BaseMetric):
    """
    Measures the growth in creativity between the first and last user answers
    in a conversation, using an LLM judge.
    """

    def __init__(self, num_retries: int = 50, retry_after: int = 60, **llm_kwargs):
        self.num_retries = num_retries
        self.retry_after = retry_after
        # Prefer DeepSeek hosted model if configured in model_config to avoid Anthropic API key dependency.
        try:
            from model_config import (
                API_BASE_deepseek,
                API_KEY_deepseek,
                LITELLM_MODEL_NAME_deepseek,
            )
            default_model = LITELLM_MODEL_NAME_deepseek
            self.llm_kwargs: Dict[str, Any] = {
                "temperature": 0.0,
                "model": default_model,
                "api_base": API_BASE_deepseek,
                "api_key": API_KEY_deepseek,
                **llm_kwargs,
            }
        except Exception:
            # Fallback to Claude if DeepSeek config missing
            self.llm_kwargs: Dict[str, Any] = {
                "temperature": 0.0,
                "model": "claude-3-5-sonnet-latest",
                **llm_kwargs,
            }

    def _score_answer(self, answer_text: str) -> Optional[float]:
        prompt = CREATIVITY_PROMPT.format(answer_text=answer_text)
        for i in range(self.num_retries):
            try:
                full_response = litellm.completion(
                    **self.llm_kwargs,
                    messages=[{"role": "user", "content": prompt}],
                    num_retries=1
                ).choices[0].message.content
                logger.info(f"[CreativityMetric] LLM response received (retry {i+1})")
            except Exception as e:
                import time
                time.sleep(self.retry_after)
                logger.error(f"[retry={i + 1}] Error during LLM call: {e}")
                continue

            try:
                if isinstance(full_response, str):
                    full_response = extract_json(full_response)
            except Exception as e:
                logger.error(f"Error extracting JSON: {e}")
                logger.debug("LLM raw response before extraction: %s", full_response)
                continue

            # Debug: log the parsed response keys/values
            if isinstance(full_response, dict):
                logger.debug("Parsed LLM response keys: %s", list(full_response.keys()))
                if 'creativity' in full_response:
                    logger.debug("Parsed creativity value: %s", full_response.get('creativity'))
                    return full_response['creativity']
                else:
                    logger.debug("Parsed response has no 'creativity' key; full_response=%s", full_response)

        logger.warning(f"Failed to get valid creativity score after {self.num_retries} retries")
        return None

    def score(
        self,
        prompt: str,
        groundtruth: str,
        completion: Optional[str],
        messages: Optional[List[Dict[str, str]]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> float:
        if not messages:
            raise ValueError("`messages` must be provided for GrowthCreativityMetric.")

        user_msgs = [m["content"] for m in messages if m["role"] == "user"]
        # if len(user_msgs) < 2:
        #     # Not enough user messages to measure growth
        #     return 0.0

        first_score = self._score_answer(user_msgs[0])
        #last_score = self._score_answer(user_msgs[-1])

        # if first_score is None or last_score is None:
        #     return 0.0

        if first_score is None:
            logger.warning("Failed to get creativity score from LLM, returning 0.0")
            return 0.0
        
        return first_score / 5
