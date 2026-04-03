
# AlphaContext: An Evolutionary Tree-based Psychometric Context Generator for Creativity Assessment (Ongoning Work)



## Overview

AlphaContext provides an end-to-end pipeline for generating creativity-assessment contexts from title–theme inputs: it plans an outline, generates structured text, and applies MAP-Elites to optimize both quality and diversity, producing assessment-ready contexts.

### Code mapping

- **Pipeline entry / orchestration**
  - `generate.py` — runs the full end-to-end workflow 

- **(1) HyperTree Outline Planner**
  - `modules/htp_outline.py` — hypertree/outline planning implementation

- **(2) MCTS-based Context Generator**
  - `modules/mcts.py` — MCTS sentence-level generation/search module

- **(3) Evolutionary optimization (MAP-Elites)**
  - `modules/map_elites_text.py` — MAP-Elites text optimizer

---

## Installation

This project is tested on **Linux** with **Python 3.9**.

To get started, create a new environment and install requirements via pip:
```bash
conda create -n AlphaContext python=3.9 -y
conda activate AlphaContext
python -m pip install -U pip
python -m pip install -r requirements.txt
```
On first run, the script may automatically download `en_core_web_sm` if it is missing.
---

## Quick Start

Run the full pipeline:

```bash
python generate.py
```

By default, it reads inputs from `input/` and writes results to `outputs/`.

---

## Input

Default input directory: `input/`

* Expected format: JSON array. Each entry should be an object with `title`, and `theme` fields. Example:

```json
[  
  { "title": "AI Partner", "theme": "Human–AI companionship and autonomy: ethics, emotional reliance, privacy, and governance of pervasive personal AI assistants." },
  { "title": "Ocean Soup Future Scene", "theme": "Marine plastics futures: circular economy, cleanup tech, ecosystem impacts, and policy for ocean health." }
]
```

* Default input examples: `input/test_dataset.json`.

---

## Output

Default output directory: `outputs/`

* Typical output layout / example artifacts produced by scripts:
  - `outputs/final_context_*.json` and `outputs/final_contexts/final_contexts_all.json` — generated final contexts.
  - `outputs/assembled_for_map_*.json` — assembled inputs prepared for MAP-Elites runs.
  - `outputs/final_contexts_scored_*.json` — scored contexts with metadata.
  - `outputs/map_elites_text_model_run/` — MAP-Elites run artifacts and archives.
  - `outputs/multiturn_data/` — persona-aware multiturn scoring outputs.
  - `outputs/prompt_generated/` — intermediate generated prompts and drafts.

---

## Model Configuration

All model endpoints are configured in `model_config.py` .

Edit the following fields:

- `MODEL_NAME_deepseek`: model identifier used by the scripts (e.g., `DeepSeek-V3.1`).
- `API_BASE_deepseek`: base URL of your model service (ending with `/v1` is recommended).
- `CHAT_EP_deepseek`: derived chat endpoint (`/chat/completions`).
- `COMP_EP_deepseek`: derived completions endpoint (`/completions`).

Example (`model_config.py`):

```python
MODEL_NAME_deepseek = "DeepSeek-V3.1"
API_BASE_deepseek = "https://<YOUR_HOST>/v1"
CHAT_EP_deepseek = API_BASE_deepseek.rstrip("/") + "/chat/completions"
COMP_EP_deepseek = API_BASE_deepseek.rstrip("/") + "/completions"
```

## License

MIT License. See `LICENSE`.
