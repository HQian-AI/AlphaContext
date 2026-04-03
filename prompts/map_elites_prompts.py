# Prompts for MAP-Elites text generation and evaluation

# System prompt for fitness evaluation (Coherence, Relevance, Engagement)
SYSTEM_PROMPT_FITNESS_EVALUATION = (
    "You are a scoring assistant specialized in evaluating problem contexts for creative challenges. Return a single JSON object with numeric fields: Coherence, Relevance, Engagement."
)

FITNESS_EVALUATION_PROMPT = (
    "Analyze the following problem context and return a JSON object with numeric scores (0..1, float) for the keys exactly:\n"
    "- Coherence: The problem context is logically consistent and smoothly structured: it clearly anchors time, place, and scope, lets a single central creative challenge emerge naturally from the situation, and links background, events, and constraints without confusing jumps or irrelevant information.\n"
    "- Relevance: The problem context stays tightly aligned with the assigned theme and central creative challenge: most sentences provide specific information, constraints, or stakeholder viewpoints that shape the creative problem space and support people's idea generation, rather than introducing details that have no real impact on how the problem can be understood or solved.\n"
    "- Engagement: The problem context is engaging and motivational: it presents the unresolved creative challenge as meaningful and intriguing, and clearly positions the reader as an active problem solver who is invited to explore possibilities, weigh trade-offs, and imagine alternative solutions.\n"
    "Return ONLY the JSON object, e.g. {\"Coherence\":0.75, \"Relevance\":0.8, \"Engagement\":0.6}\nText:\n"
)

# System prompt for phenotype evaluation (student_relevance, evidence_anchoring, stakeholder_breadth)
SYSTEM_PROMPT_PHENOTYPE_EVALUATION = (
    "You are a scoring assistant. Return a single JSON object with numeric fields: student_relevance, evidence_anchoring, stakeholder_breadth."
)

PHENOTYPE_EVALUATION_PROMPT = (
    "Analyze the following text and return a JSON object with numeric scores (0..1, float) for the keys exactly:"
    "\n- student_relevance: 0 = fully personal/campus, 1 = fully public/social."
    "\n- evidence_anchoring: 0 = none, 1 = strong data/mechanistic grounding."
    "\n- stakeholder_breadth: 0 = single actor, 1 = many diverse stakeholders."
    "\nReturn ONLY the JSON object, e.g. {\"student_relevance\":0.2, \"evidence_anchoring\":0.3, \"stakeholder_breadth\":0.1}\nText:\n"
)

# System prompt for mutation
SYSTEM_PROMPT_MUTATION = (
    "You are a creative rewriting assistant. Given an initial problem context text,"
    " produce a new version that moves the text toward the requested targets on the three behavior axes:"
    " student_relevance, evidence_anchoring, and stakeholder_breadth."
)

MUTATION_PROMPT_TEMPLATE = (
    "Please mutate the following text to create a new version that moves toward the target values on the three behavior axes: student_relevance, evidence_anchoring, and stakeholder_breadth.\n\n"
    "GOAL\n"
    "Produce a revised context that BOTH:\n"
    "(1) moves toward the target buckets on the three behavior axes, and\n"
    "(2) improves quality — especially coherence (problem-context consistency, clear structure), fluency (sentence rhythm, readability, natural transitions), and creativity.\n"
    "Use bold, high-impact edits throughout the text.\n"
    "You may perform multiple operations (Add/Delete/Modify) as needed.\n"
    "If needed, add bridge sentences for coherence.\n\n"
    "TARGET LENGTH\n"
    "- Aim for approximately 1000 words.\n\n"
    "TARGET VALUES (floats 0.0 to 1.0)\n"
    "- Student Relevance = {}      # 0.0 = fully personal/campus, 1.0 = fully public/social\n"
    "- Evidence Anchoring = {}      # 0.0 = none, 1.0 = strong data/mechanistic grounding\n"
    "- Stakeholder Breadth = {}          # 0.0 = single primary stakeholder, 1.0 = many diverse stakeholder groups\n"
    "AXIS TACTICS (use aggressively to reach targets)\n"
    "- Relevance-Shift:\n"
    "  • Toward 0.0 (personal/campus): replace/add anchor entities with \"classmates/course/club/family/campus activity\".\n"
    "  • Toward 1.0 (social/public): replace/add anchor entities with \"community/policy/industry/media/statistics\".\n"
    "- Evidence-Inject / Evidence-Trim:\n"
    "  • Toward higher (→0.5-1.0): add quantitative facts (value/ratio/trend) OR mechanistic explanation + generic source note\n"
    "    (\"school survey/local statistics/report/club records\").\n"
    "  • Toward lower (→0.0-0.5): remove weak stats/jargon; keep clear reasoning.\n"
    "- Stakeholder-Expand / Stakeholder-Narrow:\n"
    "  • Toward broader (→1.0): add multiple stakeholder groups with distinct incentives (e.g., residents, schools, NGOs, firms, regulators, media), and make their interactions matter.\n"
    "  • Toward narrower (→0.0): collapse to one primary decision-maker + one counterpart; reduce institutions/coalitions; keep consequences local to the immediate parties.\n\n"
    "HARD CONSTRAINTS\n"
    "- Mutate the seed: preserve global facts: named entities, roles, timeline, and causal relations.\n"
    "- Keep POV and tense unchanged.\n"
    "- Length change within ±10% of original.\n"
    "- If adding a bridge sentence, use a discourse marker: \"Therefore/Meanwhile/However/Subsequently/In summary\".\n\n"
    "ENGAGEMENT (gentle guidance)\n"
    "- Make the reader care by clarifying what is at stake and why it matters in this situation.\n"
    "- Let tension arise from realistic constraints and trade-offs already implied by the scenario, not from dramatic language.\n"
    "- Keep the challenge active and forward-moving: each paragraph should either raise a new consideration, reveal a friction, or narrow the decision space.\n\n"
    "QUALITY HINTS\n"
    "\n"
    "- Greatly enhance coherence: ensure the problem context is consistent and smoothly structured; also improve local fluency: smooth, natural sentence flow and logical transitions.\n"
    "- Pay close attention to grammar, punctuation, and readability.\n"
    "- Improve local coherence (add bridge sentences if needed).\n"
    "- Improve creativity via concrete, plausible details or insightful connections.\n"
    "- Improve engagement by adding pressure, escalation, and a dilemma — NOT by adding filler.\n"
    "- Be bolder in mutation: emphasize shifts toward target buckets. Don't hold back, but stay plausible.\n\n"
    "OUTPUT (return the entire revised text directly, no JSON, no extra formatting):\n\n"
    "Text to mutate:\n"
)