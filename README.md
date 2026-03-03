# CoE: Training-free Multimodal Summarization via Chain-of-Events

This repository contains the official implementation of "Cut to the Chase: Training-free Multimodal Summarization via Chain-of-Events" (Accepted as CVPR 2026).

![CoE](./figures/framework.png)


---

## 🌟 1. Overview

CoE tackles long-video multimodal summarization by explicitly structuring the generation process around **events** and **cross-modal grounding**.  
Instead of end-to-end finetuning, CoE is **training-free** and focuses on **robust reasoning + evidence alignment**.


✅ **Key properties**

* Training-free & plug-and-play across datasets/domains
* Event-centric structure improves coherence and coverage
* Cross-modal grounding reduces hallucination and strengthens faithfulness

---

## 🧠 2. What CoE Contributes

🌟 **CoE contributes an event-driven, training-free MMS framework** that you can apply to multiple datasets with unified data access.

**Core contributions**

* 🧩 **Chain-of-Events reasoning**: organizes summaries around event progression rather than isolated captions.
* 🔎 **Cross-modal evidence grounding**: aligns textual claims to video evidence for improved factuality.
* 🧠 **Reason-then-write**: decouples content reasoning from style realization for better generalization.

---

## 📊 3. Performance Highlights

📈 CoE is evaluated on **8 MMS datasets** using widely adopted metrics:

* **Lexical**: BLEU-4, ROUGE
* **Semantic / Consensus**: METEOR, CIDEr, BERTScore
* **Factual / Entity**: entity-level F1 (grounding-oriented)

---

## 🗂️ 4. Data

### 🧾 4.1 The 8 Datasets

Below are the eight benchmarks used in the paper (covering **news / academic / sports / entertainment / livestream**).

| Dataset                 | Domain             | Typical Input                | Output                           |
| ----------------------- | ------------------ | ---------------------------- | -------------------------------- |
| **VIEWS**               | News               | video + transcript/article   | news-style summary               |
| **MM-AVS**              | News               | video + multimodal context   | concise summary                  |
| **XMSMO-News**          | News               | long video + text            | ultra-short TL;DW                |
| **TIB**                 | Lecture            | lecture video + transcript   | lecture summary                  |
| **VISTA**               | Academic           | talk video + transcript      | abstract-style summary           |
| **BLiSS**               | Livestream         | long livestream + transcript | process-oriented summary         |
| **SoccerNet-Caption** | Sports             | match video                  | event-centric commentary/summary |
| **SummScreen3D** | TV / Entertainment | episode + dialogue           | story-level recap                |



---

### 🗄️ 4.2 Unified MongoDB Storage

To make dataset handling clean and scalable, we normalize the **textual side** across datasets and store all examples in **MongoDB** with a consistent schema.
This enables CoE to **query different datasets in a unified way** and simplifies evaluation.

✅ Recommended MongoDB schema

| Field        | Type   | Description                                      |
| ------------ | ------ | ------------------------------------------------ |
| `dataset`    | string | dataset name (e.g., `VIEWS`)                     |
| `video_id`   | string | unique sample id                                 |
| `video_path` | string | path to video (or feature path)                  |
| `text`       | string | unified text input (transcript/article/dialogue) |
| `reference`  | string | ground-truth summary (if available)              |
| `meta`       | object | optional metadata (title, timestamps, url, etc.) |

📌 Minimal example document

```json
{
  "video_id": "000123",
  "video_path": "/data/views/videos/000123.mp4",
  "text": "Full transcript or article text here...",
  "reference": "Ground-truth summary here...",
  "meta": {"title": "News title", "date": "2025-01-01"}
}
```

---

## 🧰 5. Installation

### ✅ 5.1 Environment

* Python **3.10+**
* MongoDB (local or remote)
* (Optional) CUDA + PyTorch for acceleration

### 📦 5.2 Setup

```bash
git clone https://github.com/youxiaoxing/CoE.git
cd CoE

conda create -n coe python=3.10 -y
conda activate coe

pip install -U pip
pip install numpy tqdm pillow pymongo requests
```

### 🧠 5.3 Optional dependencies (recommended)

```bash
# if you evaluate BERTScore
pip install bert-score

# if you use torch-based backbones
pip install torch

# if you use OpenAI-style chat completion client (or compatible)
pip install openai
```



---

## 🚀 6. Inference

CoE inference is driven by **`CoE/CoE.py`**.

### 🧾 6.1 Prepare a config file (example)

Create `config.json`:

```json
{
  "mongo": {
    "uri": "mongodb://localhost:27017",
    "db": "coe"
  },
  "model": {
    "base_url": "YOUR_API_BASE",
    "api_key": "YOUR_API_KEY",
    "model_name": "YOUR_MODEL_NAME"
  },
  "processing": {
    "max_workers": 8,
    "max_segments": 12,
    "quest_eval_iterations": 2
  },
  "datasets": {
    "VIEWS": {
      "collection": "views",
      "save_file": "outputs/views.jsonl",
      "prompt_type": "news"
    }
  }
}
```

### ▶️ 6.2 Run inference

```bash
python CoE/CoE.py --config config.json --dataset VIEWS
```

📌 Output will be written to the `save_file` specified in config (e.g., `outputs/views.jsonl`).

---

## ✅ 7. Evaluation

We provide two evaluation entrypoints (typical setup):

### 🧪 7.1 Semantic scoring service (optional)

If you run BERTScore in a service mode:

```bash
python Evaluation/bert_score_server.py --host 0.0.0.0 --port 8000
```

📌 Measures: **semantic similarity** between generated summaries and references (BERTScore).

---

### 📏 7.2 Summarization metric evaluation

Run the evaluator on predictions vs references:

```bash
python Evaluation/evaluate.py \
  --pred outputs/views.jsonl \
  --ref  data/views_reference.jsonl \
  --metrics bleu rouge cider meteor bertscore entity_f1
```

📌 Measures:

* **BLEU-4 / ROUGE**: lexical overlap
* **CIDEr / METEOR**: consensus + content quality
* **BERTScore**: semantic similarity
* **entity_f1**: grounding-related factuality proxy

---

## 📬 8. Contact

For questions, issues, or collaborations:

* 👤 Xiaoxing You — `youxiaoxing@hdu.edu.cn`
* 🧑‍💻 Open an issue: **GitHub Issues** tab