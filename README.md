# CoE: Training-free Multimodal Summarization via Chain-of-Events

Official implementation of **"Cut to the Chase: Training-free Multimodal Summarization via Chain-of-Events"** (CVPR 2026).

![CoE framework](./figures/framework.png)

---

## TL;DR

CoE is a **training-free** multimodal summarization framework for long videos. It structures generation around:

- **Chain-of-Events reasoning** (event progression over isolated clips)
- **Cross-modal grounding** (summary claims linked to visual/text evidence)
- **Reason-then-write generation** (content planning before style finalization)

---

## Repository Structure

```text
CoE/
├── CoE/
│   ├── CoE.py                     # Main inference pipeline
│   ├── EventGraph.py              # Graph data structure helpers
│   ├── Jsonl_to_Json.py           # Convert raw jsonl outputs to eval-ready json
│   └── config/
│       └── Qwen_7b_config.json    # Example end-to-end config
├── Graph_Construct/
│   ├── graph_construction.py      # Event graph extraction pipeline
│   └── config/
│       └── graph_construction_config.json
├── Evaluation/
│   ├── compute_score.py           # BLEU/ROUGE/CIDEr/METEOR/BERTScore
│   └── compute_score_entity.py    # Entity-level factual evaluation
├── figures/
│   └── framework.png
└── requirements.txt
```

---

## Features

- ✅ **Training-free** pipeline (no fine-tuning required)
- ✅ Unified dataset access via **MongoDB**
- ✅ Multi-dataset config support in one codebase
- ✅ Event graph construction + inference + evaluation scripts

---

## 1) Installation

### Requirements

- Python **3.12+**
- MongoDB (local or remote)
- OpenAI-compatible inference endpoint (e.g., vLLM API)

### Environment Setup

```bash
git clone https://github.com/youxiaoxing/CoE.git
cd CoE

conda create -n coe python=3.12 -y
conda activate coe
pip install -r requirements.txt
```

> `requirements.txt` is currently an environment-export style file. If your environment differs, install missing packages on demand based on script imports.

---

## 2) Data Format (MongoDB)

CoE expects per-sample documents in a unified schema:

| Field        | Type   | Description                                      |
| ------------ | ------ | ------------------------------------------------ |
| `video_id`   | string | Unique sample id                                 |
| `video_path` | string | Relative or absolute video path                  |
| `text`       | string | Unified text input (transcript/article/dialogue) |
| `reference`  | string | Ground-truth summary (if available)              |
| `meta`       | object | Optional metadata                                |

Example:

```json
{
  "video_id": "000123",
  "video_path": "/data/views/videos/000123.mp4",
  "text": "Full transcript or article text here...",
  "reference": "Ground-truth summary here...",
  "meta": {
    "title": "News title",
    "date": "2025-01-01"
  }
}
```

---

## 3) Configure CoE

Use `CoE/config/Qwen_7b_config.json` as a template.

### Minimal config example

```json
{
  "hf_endpoint": "https://hf-mirror.com",
  "mongo": {
    "host": "Mongodb_IP_Address",
    "port": 27017,
    "database": "mms"
  },
  "model": {
    "clients": [
      "http://IP_Address:Port/v1"
    ],
    "api_key": "-",
    "model_name": "Qwen2.5-VL-7B-Instruct",
    "max_tokens": 500,
    "temperature": 0.1,
    "current_client_idx": 0
  },
  "processing": {
    "max_workers": 4,
    "max_segments": 5,
    "max_num_frames": 72,
    "control_max_frames": true,
    "frames_per_group": 6,
    "quest_eval_iterations": 3,
    "f1_threshold": 0.75
  },
  "datasets": {
    "views": {
      "collection": "views",
      "query": {
        "graph": {
          "$exists": true
        },
        "split": "test"
      },
      "video_path_template": "video_dataset/views_video_data/{split}/{video_id}.mp4",
      "save_file": "./result/views.jsonl",
      "json_save_file": "./result/views.json",
      "prompt_type": "views",
      "article_field": "storyline",
      "summary_key": "summary"
    }
  },
  "prompts": {
    "views": {
      "subevent_match": "...",
      "entity_match": "...",
      "summary_generation": "...",
      "translate_style": "..."
    }
  }
}
```

---

## 4) Build Event Graphs (Optional but Recommended)

Before inference, generate `graph` fields in MongoDB:

```bash
python Graph_Construct/graph_construction.py \
  --config Graph_Construct/config/graph_construction_config.json \
  --dataset views
```

If `--dataset` is omitted, all configured datasets are processed.

---

## 5) Run Inference

```bash
python CoE/CoE.py --config CoE/config/Qwen_7b_config.json --dataset views
```

- Output is appended to `save_file` (JSONL), e.g. `./result/views.jsonl`.
- Each line stores `_id` and generated `response`.

---

## 6) Convert JSONL Output to Evaluation JSON

```bash
python CoE/Jsonl_to_Json.py \
  --config CoE/config/Qwen_7b_config.json \
  --dataset views
```

This creates eval-ready JSON with fields:

```json
{
  "id": "...",
  "ref_caption": "...",
  "caption": "..."
}
```

---

## 7) Evaluation

### 7.1 Standard summarization metrics

```bash
python Evaluation/compute_score.py --input_path result/
```

With BERTScore:

```bash
python Evaluation/compute_score.py --input_path result/ --use_bert_score
```

Reported metrics:

- BLEU-1/2/3/4
- ROUGE
- ROUGE-LSum
- CIDEr
- METEOR
- BERTScore (optional)

### 7.2 Entity-level factual evaluation

```bash
python Evaluation/compute_score_entity.py --input_path result/
```

Optional spaCy model override:

```bash
python Evaluation/compute_score_entity.py --input_path result/ --spacy_model en_core_web_lg
```

---

## 8) Supported Datasets in Current Config

The provided config includes entries for:

- VIEWS
- MM-AVS
- XMSMO-News
- TIB
- VISTA
- BLiSS
- SoccerNet-Caption
- SummScreen3D (configured as `hierarchical3D`)

You can adapt path templates/queries to your local storage.

---

## 9) Troubleshooting

- **`No graph found` during inference**
  - Run graph construction first, or ensure each sample has `graph` in MongoDB.
- **Video loading errors**
  - Verify `video_path_template` resolves to existing files.
- **Empty evaluation results**
  - Ensure converted JSON contains both `caption` and `ref_caption`.
- **spaCy model load failure**
  - Install requested model, or switch to an installed model via `--spacy_model`.

---

## Citation

If you use CoE in your work, please cite the paper (bibtex will be added here).

---

## Contact

For questions, issues, and collaboration:

- Open a GitHub issue in this repository.
