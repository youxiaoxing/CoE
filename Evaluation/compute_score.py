import argparse
import json
import os
import re
import types

import numpy as np
from bert_score import score
from pycocoevalcap.bleu.bleu_scorer import BleuScorer
from pycocoevalcap.cider.cider_scorer import CiderScorer
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from rouge_score import rouge_scorer
from tqdm import tqdm
from unidecode import unidecode


def process_string(text: str) -> str:
    text = text.replace("\n", " ").strip()
    text = unidecode(text)
    return text


def _stat(self, hypothesis_str, reference_list):
    hypothesis_str = hypothesis_str.replace("|||", "").replace("  ", " ")
    score_line = " ||| ".join(("SCORE", " ||| ".join(reference_list), hypothesis_str))
    score_line = score_line.replace("\n", "").replace("\r", "")
    self.meteor_p.stdin.write(f"{score_line}\n".encode())
    self.meteor_p.stdin.flush()
    return self.meteor_p.stdout.readline().decode().strip()


def cal_caption_score_from_dict(result_dict, use_bert_score: bool = False):
    bleu_scorer = BleuScorer(n=4)
    rouge_scorer_old = Rouge()
    rouge_scorer_new = rouge_scorer.RougeScorer(["rougeL", "rougeLsum"], use_stemmer=True)

    rouge_scores = []
    rouge_lsum_scores = []

    cider_scorer = CiderScorer(n=4, sigma=6.0)
    meteor_scorer = Meteor()
    meteor_scorer._stat = types.MethodType(_stat, meteor_scorer)

    eval_line = "EVAL"
    meteor_scorer.lock.acquire()
    count = 0
    meteor_scores = []

    candidates = []
    references = []

    for sample in tqdm(result_dict, desc="Processing"):
        caption = re.sub(r"[^\w\s]", "", sample["ref_caption"])
        generation = re.sub(r"[^\w\s]", "", sample["caption"])

        bleu_scorer += (generation, [caption])

        rouge_score_val = rouge_scorer_old.calc_score([generation], [caption])
        rouge_scores.append(rouge_score_val)

        rouge_score_new = rouge_scorer_new.score(sample["ref_caption"], sample["caption"])
        rouge_lsum_scores.append(rouge_score_new["rougeLsum"].fmeasure)

        cider_scorer += (generation, [caption])

        stat = meteor_scorer._stat(generation, [caption])
        eval_line += f" ||| {stat}"
        count += 1

        candidates.append(sample["caption"])
        references.append(sample["ref_caption"])

    meteor_scorer.meteor_p.stdin.write(f"{eval_line}\n".encode())
    meteor_scorer.meteor_p.stdin.flush()

    for _ in range(count):
        meteor_scores.append(float(meteor_scorer.meteor_p.stdout.readline().strip()))
    meteor_score = float(meteor_scorer.meteor_p.stdout.readline().strip())

    meteor_scorer.lock.release()

    bleu_score, _ = bleu_scorer.compute_score(option="closest")
    rouge_score = float(np.mean(np.array(rouge_scores))) if rouge_scores else 0.0
    rouge_lsum_score = float(np.mean(np.array(rouge_lsum_scores))) if rouge_lsum_scores else 0.0
    cider_score, _ = cider_scorer.compute_score()

    if use_bert_score:
        P, R, F1 = score(candidates, references, lang="en", verbose=False)
        bert_score = float(F1.mean().item())
    else:
        bert_score = 0.0

    return {
        "bleu": bleu_score,
        "rouge": rouge_score,
        "rouge_lsum": rouge_lsum_score,
        "cider": cider_score,
        "meteor": meteor_score,
        "bert_score": bert_score,
    }


def get_data_file(folder_path: str):
    json_files = sorted([f for f in os.listdir(folder_path) if f.endswith(".json")])
    print(f"\nFound {len(json_files)} JSON files\n")

    all_data = []
    file_names = []

    for json_file in json_files:
        file_path = os.path.join(folder_path, json_file)
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        data = [item for item in data if "Failed to process video" not in item.get("caption", "")]

        for item in data:
            if "</s>" in item.get("caption", ""):
                item["caption"] = process_string(item["caption"].split("</s>")[0])
            item["ref_caption"] = process_string(item.get("ref_caption", "").replace("</s>", ""))

        all_data.append(data)
        file_names.append(json_file)

    return all_data, file_names


def print_results(results_dict):
    print("\n" + "=" * 120)
    print(f"{'Evaluation Results':^120}")
    print("=" * 120 + "\n")

    for file_name, scores in results_dict.items():
        print(f"File: {file_name}")
        print("-" * 120)
        print(f"  BLEU-1: {scores['bleu'][0]:.4f}")
        print(f"  BLEU-2: {scores['bleu'][1]:.4f}")
        print(f"  BLEU-3: {scores['bleu'][2]:.4f}")
        print(f"  BLEU-4: {scores['bleu'][3]:.4f}")
        print(f"  ROUGE:  {scores['rouge']:.4f}")
        print(f"  ROUGE-LSum: {scores['rouge_lsum']:.4f}")
        print(f"  CIDEr:  {scores['cider']:.4f}")
        print(f"  METEOR: {scores['meteor']:.4f}")
        print(f"  BERTScore: {scores['bert_score']:.4f}")
        print()

    print("=" * 120 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Caption evaluation script")
    parser.add_argument(
        "--input_path",
        "-i",
        type=str,
        required=True,
        help="Path to a folder containing JSON files to evaluate",
    )
    parser.add_argument(
        "--use_bert_score",
        "-b",
        action="store_true",
        help="Whether to compute BERTScore (default: False)",
    )
    args = parser.parse_args()

    if not os.path.exists(args.input_path):
        print(f"Error: path does not exist - {args.input_path}")
        return
    if not os.path.isdir(args.input_path):
        print(f"Error: not a directory - {args.input_path}")
        return

    print(f"Input folder: {args.input_path}")
    print(f"Use BERTScore: {args.use_bert_score}")

    all_data, file_names = get_data_file(args.input_path)

    results_dict = {}
    for data, file_name in zip(all_data, file_names):
        print(f"Evaluating: {file_name} (N={len(data)})")
        scores = cal_caption_score_from_dict(data, use_bert_score=args.use_bert_score)
        results_dict[file_name] = scores

    print_results(results_dict)


if __name__ == "__main__":
    main()