from typing import Dict, Any
# Per-part prompt templates and post-processing rules
PART_TEMPLATES: Dict[str, Dict[str, Any]] = {
    "Anchor": {
        "prompt": (
            "Context:  {context}\n\n"
            "Describe the time, location, and background: Set a concrete time and societal context. The background should highlight societal issues, environmental changes, resource pressures, etc., making the context feel grounded and realistic. Introduce the character and the environment in a way that highlights their current emotional or existential struggles. Under this goal, generate the opening paragraph to initiate the scenario, the Anchor section of the stimulus text."
            "Requirements: The opening should naturally introduce the character and their surroundings, presenting the challenges they face in a realistic, human way, without making technology or machines the central focus."
            "Task:  Generate {k} context fragments for this section (Anchor(Time + Location + Scale)) – each should preferably be a single, short sentence, with a target length of approximately {max_chars} characters. These fragments will serve as the opening of the context, and are required to set the tone, introduce characters/situations, and be expressive.\n"
            "Strict Requirements: \n"
            "  - Return only a JSON array (e.g.:  [\"Sentence 1\", \"Sentence 2\"]), with no additional explanations or meta-tags.\n"
            "  - Output Requirement: This part including the precise year, specific location (city/region/facility), and the scale of the scene (community / national / international / space)."
            "  - Each element in the array must be a complete sentence (it should have core components such as a subject and predicate, be able to convey a complete meaning independently, and must not be an incomplete sentence or a statement with missing components; do not use placeholders or other tags either).\n"+
            "  - Ensure that readers can understand a clear scene, action, or state just by reading the sentence alone."
            "\nExample Output: [\"In 2045, as climate change continued to worsen, the ‘Green Haven’ city in the western region became one of the last places humans could still live.\"]"
        ),
        "max_chars": 1024,
        "min_chars": 100,
        "candidates": 2,
        "target_sentences": 3,
        "max_tokens": 1024,
        "temperature": 0.6,
    },
    "Scene Setting": {
        "prompt": (
            "Context:  {context}\n\n"
            "Depict a realistic everyday background, embedding a long-term trend or external pressure (technology, ecology, economy, social rules)."
            "Include concrete objects / action details (e.g., devices, routines, systems) and 1–3 potential sources of conflict, chosen from: Arts & Aesthetics, Basic Needs, Business & Commerce, Communication, Culture & Religion, Defense, Economics, Education, Environment, Ethics & Morality, Government & Politics, Law & Justice, Miscellaneous, Physical Health, Psychological Health, Recreation, Science, Social Relationships, Technology, Transportation."
            "Requirements: Use concrete details to show societal background and character interactions, where challenges gradually unfold through their daily lives."
            "Task: Generate {k} context fragments for this section (Scene Setting(Environment + Daily Activities + Background Trends)) – each should preferably be a single, short sentence, with a target length of approximately {max_chars} characters. The fragments are required to introduce a triggering event/turning point, creating tension or a change in the context.\n"
            "Strict Requirements: "
            "  - Return only a JSON array, and each element must be a complete sentence; do not use any meta-tags or additional explanations."
            "  - Output Requirement: Include concrete objects / action details (e.g., devices, routines, systems) and 1–3 potential sources of conflict, chosen from: Arts & Aesthetics, Basic Needs, Business & Commerce, Communication, Culture & Religion, Defense, Economics, Education, Environment, Ethics & Morality, Government & Politics, Law & Justice, Miscellaneous, Physical Health, Psychological Health, Recreation, Science, Social Relationships, Technology, Transportation."
            "  - Each element in the array must be a complete sentence (it should have core components such as a subject and predicate, be able to convey a complete meaning independently, and must not be an incomplete sentence or a statement with missing components; do not use placeholders or other tags either).\n"
            "  - Ensure that readers can understand a clear scene, action, or state just by reading the sentence alone."
            "\nExample Output: [\"On Saturday morning, Li Na received an urgent notice about an emergency remote meeting.\"]"
        ),
        "max_chars": 2048,
        "min_chars": 200,
        "candidates": 2,
        "target_sentences": 5,
        "max_tokens": 2048,
        "temperature": 0.7,
    },
    "Characters & Interaction": {
        "prompt": (
            "Context:  {context}\n\n"
            "Character interactions and emotional development: Use dialogue and behavior to reveal how characters respond to their inner conflicts, social dilemmas, and emotional challenges. The characters should be multi-dimensional, displaying a range of emotions and psychological depth.\n"
            "Task: Generate {k} context fragments for this section (Characters & Interaction) – each should preferably be a single, short sentence, with a target length of approximately {max_chars} characters.\n"
            "Strict Requirements: "
            "  - Return only a JSON array, and each element must be a complete sentence; do not use any meta-tags or additional explanations."
            "  - Output Requirement: Write as short dialogue or action description, including at least one explicit “problem sentence” (key context challenge expressed as a question or debate, without directly saying “challenge”). The selected challenge category should be chosen from the list: Arts & Aesthetics, Basic Needs, Business & Commerce, Communication, Culture & Religion, Defense, Economics, Education, Environment, Ethics & Morality, Government & Politics, Law & Justice, Miscellaneous, Physical Health, Psychological Health, Recreation, Science, Social Relationships, Technology, Transportation."
            "  - Each element in the array must be a complete sentence; do not use placeholders.\n"
            "\nExample Output: [\"A young resident asked, ‘Why is our food quota 20% less than last year?’\"]"
        ),
        "max_chars": 1024,
        "min_chars": 260,
        "candidates": 2,
        "target_sentences": 7,
        "max_tokens": 2048,
        "temperature": 0.75,
    },
    "Conflict & Challenge": {
        "prompt": (
            "Context:  {context}\n\n"
            "Conflict presentation: Present societal and personal conflicts, especially those involving inner struggles, ethical dilemmas, and social pressures. These conflicts should emerge naturally from the characters' daily lives and decisions, rather than being directly explained."
            "Task: Generate {k} context fragments for this section (Conflict & Challenge) – each should preferably be a single, short sentence, with a target length of approximately {max_chars} characters.\n"
            "Strict Requirements: "
            "  - Return only a JSON array, and each element must be a complete sentence; do not use any meta-tags or additional explanations."
            "  - Output Requirement: Write as short dialogue or action description, including at least one explicit “problem sentence” (key context Conflict & Challenge expressed as a question or debate, without directly saying “Conflict & Challenge”). The selected Conflict & Challenge category should be chosen from the list: Arts & Aesthetics, Basic Needs, Business & Commerce, Communication, Culture & Religion, Defense, Economics, Education, Environment, Ethics & Morality, Government & Politics, Law & Justice, Miscellaneous, Physical Health, Psychological Health, Recreation, Science, Social Relationships, Technology, Transportation."
            "  - Each element in the array must be a complete sentence; do not use placeholders.\n"
            "\nExample Output: [\"‘How long can we keep this up without breaking someone?’ Maria asked, staring at the ration ledger.\" ]"
        ),
        "max_chars": 1024,
        "min_chars": 300,
        "candidates": 2,
        "target_sentences": 6,
        "max_tokens": 2048,
        "temperature": 0.7,
    },
    "Open Task": {
        "prompt": (
            "Context:  {context}\n\n"
            "Based on the provided context text {context}, Generate ONE Open Task question that asks the student to identify multiple challenges implied in the context. Do not list challenges.\n"
            "Strict Requirements:\n"
            "1. The question MUST start with 'Please apply the problem-solving process to analyze [specific theme from the context] and identify challenges.'\n"
            "2. Please summarize the '[theme]' part of the context and generate the question accordingly.\n"
            "3. Format: Return ONLY a JSON array with ONE sentence. No other text.\n"
            "4. The sentence MUST follow this exact structure:\n"
            "'Please apply the problem-solving process to analyze [theme] and identify challenges.'\n"
            "\nExample Output Format: [\"Please apply the problem-solving process to analyze [theme] and identify multiple challenges.\"]\n"
            "\nNOTE: Do NOT generate additional context content - output ONLY the analysis question."
        ),
        "max_chars": 200,
        "min_chars": 150,
        "candidates": 2,
        "target_sentences": 1,
        "max_tokens": 180,
        "temperature": 0.3,
    },
}


# System prompt for generation
SYSTEM_PROMPT_GENERATION = (
     "You are a creativity assessment and scenario design expert with exceptional contexttelling abilities. Your task is to generate highly coherent and engaging scenario-based texts that naturally extend from the given context. Pay special attention to:\n\n"
                "1. Narrative Coherence & Flow:\n"
                "- **Logical Consistency & Smooth Structure**: Ensure the entire scenario follows a rigorous logical chain, with no confusing jumps, contradictions, or irrelevant information.\n"
                "- **Clear Anchoring of Core Elements**: Explicitly and naturally anchor the scenario's time, place, and scope at the outset, and maintain consistency of these elements throughout the narrative.\n"
                "- **Natural Emergence of Central Creative Challenge**: Let a single, focused creative challenge arise organically from the situation and context, rather than being forced or stated directly.\n"
                "- **Seamless Linkage of Background, Events, and Constraints**: Connect the scenario's background setting, unfolding events, and inherent constraints in a smooth, interdependent manner.\n"
                "- **Smooth Transitions & Cause-Effect Relationships**: Each new sentence must flow naturally from the previous one; use logical cause-effect connections to advance the narrative.\n"
                "- **Consistent Core Elements**: Maintain unchanged characters, fixed locations, and a linear timeline throughout the scenario to avoid fragmentation.\n"
                "- **Purposeful Narrative Progression**: Every sentence must meaningfully deepen the context, advance the scenario, or reinforce the central challenge—no redundant or tangential content.\n"
                "2. Significance & Concreteness:\n"
                "- Ground the context in real-world issues with genuine social significance (environment, community, technology, health, etc.)\n"
                "- Localize with concrete details: specific times, places, people, and circumstances so readers form vivid mental pictures\n"
                "- Make the creative challenge feel valuable and worth pursuing\n\n"
                "3. Context Integration & Quality:\n"
                "- Carefully read and understand the existing context, reference previously mentioned elements\n"
                "- Add new details that enrich rather than contradict; maintain thematic focus\n"
                "- Create vivid imagery with knowledge-supported contexts; include meaningful stakes and genuine uncertainty\n\n"
                "The title and theme of the context is: {title}, {theme}\n"
                "Do not output any procedural thinking, brainstorming, explanations, or other meta information(such as '<think>', 'I need', 'Let's', 'First', 'Brainstorm', etc.). "
                "If you cannot meet the requirement, return only an empty array [].Example output:  [\"A fragment.\"]"
                "Maintain narrative continuity across sections {{Anchor, Scene Setting, Characters & Interaction, Conflict & Challenge, Open Task}}:Each new sentence should build naturally upon the previous one, maintaining continuity in time, location, and characters, while gradually expanding or deepening the scene."
                "Each new paragraph must follow naturally from the previous one, preserving continuity of time, location, and characters, and progressively extending or deepening the scene and its causal state."
) 

# Evaluation prompt
EVALUATION_PROMPT = """You are a strict evaluator.

You will score the CANDIDATE SENTENCE (Fragment) using 4 dimensions on a 1–5 Likert scale,
    then convert to 0–1 by: (likert - 1) / 4.
    Return ONLY a JSON object with four float scores in [0,1]:
    {{"scaffolding_and_adherence": float, "image": float, "coherence": float, "hallucination": float}}

    === Full Text (for coherence and hallucination) ===
    {full_text}

    === Module Text (for scaffolding_and_adherence and image) ===
    {module_text}

    SCORING RULES:
    (Note: For scaffolding_and_adherence and image, evaluate only the Module Text. For coherence and hallucination, evaluate the Full Text.)

            1) Scaffolding and Adherence (Outline-grounded)
            Rate how well the text simultaneously achieves:
            (A) Implicit challenge scaffolding: it subtly plants meaningful problem cues that invite creative reasoning (trade-offs, constraints, stakeholder tensions, second-order effects), without explicitly listing "the challenges are...".
            (B) Outline adherence: these implicit cues faithfully follow the outline hints above ({outline_hints})—including required themes, keywords, constraints, and intended challenge categories.

            What to look for (high score requires most of these):
            - Implicit-but-actionable cues: understated yet concrete enough to support reasoning and solution exploration.
            - Traceable category alignment: most cues map naturally to the intended challenge categories in {outline_hints}.
            - Coverage & emphasis: reflects the outline’s priorities/constraints (not necessarily exhaustive, but clearly aligned).

            Likert anchors:
            - 1: Little/no implicit challenge scaffolding AND/OR clear deviation from {outline_hints}; cues are missing or generic.
            - 3: Some implicit cues and partial alignment, but cues are shallow/vague, coverage is uneven, or noticeable drift exists.
            - 5: Multiple high-quality implicit cues that are easy to reason about, with clear, traceable adherence to {outline_hints}, and minimal drift.

            2) Image (Engaging Imagery & Immersion)
            Rate how strongly the text pulls the reader into the scene and sustains engagement—by presenting the unresolved creative challenge as meaningful and intriguing, clearly positioning the reader as an active problem solver invited to explore possibilities, weigh trade-offs, and imagine alternative solutions—through concrete, sensory, and situational detail—so it becomes easy to picture and mentally “sticky” (curiosity, tension, stakes), without vague adjectives or empty hype.

            What to look for (high score requires most of these):

            - Situational anchors: clear time/place/people, relevant objects/resources, and what is happening now; ties to the unresolved creative challenge.

            - Sensory concreteness: specific sights/sounds/signals; avoids generic “nice/bad” descriptors.

            - Perspective & relatability: details that help readers inhabit the roles and decisions (as active problem solvers); vivid details serve understanding/engagement.

            Likert anchors:

            - 1: Flat and generic; hard to visualize; little immersion or engagement; fails to present a meaningful unresolved challenge or position readers as active problem solvers; mostly abstract statements.

            - 3: Some vivid moments, but engagement is uneven; details feel scattered, repetitive, or low-stakes; weak presentation of the unresolved challenge and guidance for readers as problem solvers.

            - 5: Highly immersive and engaging; consistent concrete detail creates a clear “mental movie” with meaningful tension/stakes; effectively presents a meaningful unresolved creative challenge and positions readers as active problem solvers.
            
            3) Coherence (Flow & Causal Continuity)
            Rate how smoothly the fragment reads as a single, consistent micro-context or scenario: events and claims should connect by clear cause-and-effect, entities stay stable, and transitions feel natural.

            What to look for (high score requires most of these):
            - Stable frame: the setting (where/when) remains consistent, or shifts are explicitly signaled.
            - Entity consistency: the same people/organizations/objects keep the same roles, names, and attributes.
            - Causal links: actions have reasons; consequences follow; the reader can answer “why did that happen?”.
            - Temporal clarity: sequence is understandable (what happens first/next); no confusing jumps.

            Likert anchors:
            - 1: Disjointed or confusing; frequent jumps/contradictions; unclear who/where/when/why.
            - 3: Mostly understandable, but with noticeable gaps (weak causal links, minor contradictions, or abrupt transitions).
            - 5: Clear, continuous, and internally consistent; strong causal/temporal flow with natural transitions.


            4) Hallucination (Internal Support Penalty)
            This is a penalty dimension: rate the extent to which the fragment contains claims that feel unsupported, fabricated, or internally unjustified.

            What to look for (high score means worse hallucination):
            - Reality breaks: implausible leaps that violate common sense or the fragment’s established situation.
            - Self-inconsistency: details that conflict with earlier statements (names, roles, locations, constraints).
            - Overconfident assertions: definitive causal claims or outcomes that the fragment does not substantiate.

            Likert anchors:
            - 1: No hallucination; claims are plausible, restrained, and consistent with what the fragment establishes.
            - 3: Some questionable or weakly supported details, but they do not dominate the fragment.
            - 5: Many unsupported/fabricated claims or reality breaks that undermine reliability.

    Output format:
    Return ONLY a JSON object like:
    {{"scaffolding_and_adherence": 0.75, "image": 0.50, "coherence": 1.00, "hallucination": 0.25}}
    No explanations, no extra text.
"""
