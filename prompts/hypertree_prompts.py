# agents/hypertree_prompts.py
from langchain.prompts import PromptTemplate

planner_agent_prompt = PromptTemplate(
    input_variables=["title", "theme"],
    template=(
        "You are a professional planner specializing in designing and structuring future-oriented creativity problem-context outline.\n\n"
        "TITLE:\n{title}\n\n"
        "THEME:\n{theme}\n\n"
        "Task:\n"
        "- Design a coherent high-level creativity problem-context concept that fits the given title and theme.\n"
        "- Your goal is not to write the full context, but to outline its conceptual direction —\n"
        "  Identifying the narrative anchor, setting, key characters and interactions, core conflicts, and open task ideas.\n\n"
        "Guidelines:\n"
        "1) The context should reflect the theme deeply — showing cause–effect relationships, ethical or technological tensions, and human values in a future-oriented setting.\n"
        "2) The outline must be imaginative yet logically consistent (futuristic realism is preferred over pure fantasy).\n"
        "3) Avoid superficial tropes; focus on the underlying systemic or emotional challenges that can elicit rich thinking.\n"
        "4) Keep the plan structured and concise; you may use headings like [Anchor], [Scene Setting], [Characters & Interaction], [Conflict & Challenge], [Open Task].\n"
        "5) Do NOT output a full narrative or dialogue; provide only the high-level conceptual plan for the problem-context.\n"
    )
)
select_prompt = PromptTemplate(
    input_variables=["title", "theme", "current_tree", "leaves"],
    template=(
        "You are a professional planner specializing in designing and structuring creativity problem-contexts.\n\n"
        "Context Title: {title}\n"
        "Theme: {theme}\n\n"
        "Current context outline tree:\n"
        "{current_tree}\n\n"
        "The following nodes can be expanded next:\n"
        "{leaves}\n\n"
        "Please select ONE node index to expand next, based on logical contexttelling flow.\n"
        "Respond ONLY with the index number (e.g., '0' or '2')."
    )
)



seed_select_prompt = PromptTemplate(
    input_variables=["title", "theme", "full_tree", "node_name", "candidate_seeds", "min_k", "max_k"],
    template=(
        "You are a creativity assessment and scenario design expert. "
        "Your task is to select high-level challenge seeds "
        "for a future creativity problem-context.\n\n"
        "Title: {title}\n"
        "Theme: {theme}\n\n"
        "FULL HYPERTREE (for global context):\n"
        "{full_tree}\n\n"
        "Expand Node: {node_name}\n\n"
        "Selection objective:\n"
        "- Choose items with the HIGHEST relevance to (1) the Theme, (2) the FULL HYPERTREE context, "
        "and (3) the current Expand Node.\n"
        "- Prioritize node-specific relevance over generic theme-level relevance.\n\n"
        "From the challenge seeds below, choose {min_k}–{max_k} items.\n"
        "IMPORTANT: Return them sorted from MOST relevant to LEAST relevant.\n\n"
        "Candidate challenge seeds:\n"
        "{candidate_seeds}\n\n"
        "Rules:\n"
        "- Select ONLY from the candidate list.\n"
        "- Output order MUST be descending by relevance.\n"
        "- Return ONLY a JSON array of strings. No extra text.\n"
    ),
)


optional_pick_prompt = PromptTemplate(
    input_variables=[
        "title", "theme", "full_tree", "node_name",
        "candidates", "range_text", "order_rule"
    ],
    template=(
        "You are a professional planner.\n\n"
        "Title: {title}\n"
        "Theme: {theme}\n\n"
        "FULL HYPERTREE (for global context):\n"
        "{full_tree}\n\n"
        "Node to expand: {node_name}\n\n"
        "Selection objective:\n"
        "- Select children with the HIGHEST relevance to (1) the Theme, (2) the FULL HYPERTREE context, "
        "and (3) the current Expand Node.\n"
        "- Prioritize node-specific relevance over generic theme-level relevance.\n\n"
        "Candidates:\n"
        "{candidates}\n\n"
        "Task: {range_text}\n"
        "{order_rule}\n"
        "Return ONLY a JSON array of strings (items must be exactly from the candidates).\n"
    )
)
