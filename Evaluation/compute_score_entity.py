import argparse
import json
import os
from collections import defaultdict

import spacy
from tqdm import tqdm
from unidecode import unidecode


def process_string(text: str) -> str:
    text = text.replace("\n", " ").strip()
    text = unidecode(text)
    return text


def contain_entity_by_gtent(entities, target, gt_first: bool = True) -> bool:
    for ent in entities:
        if gt_first:
            if ent == target["text"]:
                return True
        else:
            if ent["text"] == target:
                return True
    return False


def get_entities(doc):
    entities = []
    for ent in doc.ents:
        entities.append(
            {
                "text": ent.text.lower(),
                "label": ent.label_,
                "tokens": [{"text": tok.text.lower(), "pos": tok.pos_} for tok in ent],
            }
        )
    return entities


def compute_entities_by_gtent(
    caption_entities,
    caption_persons,
    caption_orgs,
    caption_gpes,
    gen_entities,
    c,
):
    c["n_caption_ents"] += len(caption_entities)
    c["n_gen_ents"] += len(gen_entities)

    for ent in gen_entities:
        if contain_entity_by_gtent(caption_entities, ent, gt_first=True):
            c["n_gen_ent_matches"] += 1
    for ent in caption_entities:
        if contain_entity_by_gtent(gen_entities, ent, gt_first=False):
            c["n_caption_ent_matches"] += 1

    gen_persons = [e for e in gen_entities if e["label"] == "PERSON"]
    c["n_caption_persons"] += len(caption_persons)
    c["n_gen_persons"] += len(gen_persons)
    for ent in gen_persons:
        if contain_entity_by_gtent(caption_persons, ent, gt_first=True):
            c["n_gen_person_matches"] += 1
    for ent in caption_persons:
        if contain_entity_by_gtent(gen_persons, ent, gt_first=False):
            c["n_caption_person_matches"] += 1

    gen_orgs = [e for e in gen_entities if e["label"] == "ORG"]
    c["n_caption_orgs"] += len(caption_orgs)
    c["n_gen_orgs"] += len(gen_orgs)
    for ent in gen_orgs:
        if contain_entity_by_gtent(caption_orgs, ent, gt_first=True):
            c["n_gen_orgs_matches"] += 1
    for ent in caption_orgs:
        if contain_entity_by_gtent(gen_orgs, ent, gt_first=False):
            c["n_caption_orgs_matches"] += 1

    gen_gpes = [e for e in gen_entities if e["label"] == "GPE"]
    c["n_caption_gpes"] += len(caption_gpes)
    c["n_gen_gpes"] += len(gen_gpes)
    for ent in gen_gpes:
        if contain_entity_by_gtent(caption_gpes, ent, gt_first=True):
            c["n_gen_gpes_matches"] += 1
    for ent in caption_gpes:
        if contain_entity_by_gtent(gen_gpes, ent, gt_first=False):
            c["n_caption_gpes_matches"] += 1

    return c


def evaluate_entity_by_gtent(ref_list, gen_list, spacy_model: str = "en_core_web_lg"):
    nlp = spacy.load(spacy_model)
    ent_counter = defaultdict(int)

    for ref_caption, gen_caption in tqdm(zip(ref_list, gen_list), total=min(len(ref_list), len(gen_list))):
        gen_cap = nlp(gen_caption)
        ref_cap = nlp(ref_caption)

        gen_entities = get_entities(gen_cap)
        ref_entities = get_entities(ref_cap)

        caption_entities = [e["text"] for e in ref_entities]
        caption_persons = [e["text"] for e in ref_entities if e["label"] == "PERSON"]
        caption_orgs = [e["text"] for e in ref_entities if e["label"] == "ORG"]
        caption_gpes = [e["text"] for e in ref_entities if e["label"] == "GPE"]

        compute_entities_by_gtent(
            caption_entities,
            caption_persons,
            caption_orgs,
            caption_gpes,
            gen_entities,
            ent_counter,
        )

    def safe_div(a: int, b: int) -> float:
        return a / b if b > 0 else 0.0

    recall_all = safe_div(ent_counter["n_caption_ent_matches"], ent_counter["n_caption_ents"])
    precision_all = safe_div(ent_counter["n_gen_ent_matches"], ent_counter["n_gen_ents"])
    f1_all = (2 * precision_all * recall_all / (precision_all + recall_all)) if (precision_all + recall_all) > 0 else 0.0

    entity_results = {
        "Entity all - recall": {
            "count": ent_counter["n_caption_ent_matches"],
            "total": ent_counter["n_caption_ents"],
            "percentage": recall_all,
        },
        "Entity all - precision": {
            "count": ent_counter["n_gen_ent_matches"],
            "total": ent_counter["n_gen_ents"],
            "percentage": precision_all,
        },
        "Entity person (by full name) - recall": {
            "count": ent_counter["n_caption_person_matches"],
            "total": ent_counter["n_caption_persons"],
            "percentage": safe_div(ent_counter["n_caption_person_matches"], ent_counter["n_caption_persons"]),
        },
        "Entity person (by full name) - precision": {
            "count": ent_counter["n_gen_person_matches"],
            "total": ent_counter["n_gen_persons"],
            "percentage": safe_div(ent_counter["n_gen_person_matches"], ent_counter["n_gen_persons"]),
        },
        "Entity GPE - recall": {
            "count": ent_counter["n_caption_gpes_matches"],
            "total": ent_counter["n_caption_gpes"],
            "percentage": safe_div(ent_counter["n_caption_gpes_matches"], ent_counter["n_caption_gpes"]),
        },
        "Entity GPE - precision": {
            "count": ent_counter["n_gen_gpes_matches"],
            "total": ent_counter["n_gen_gpes"],
            "percentage": safe_div(ent_counter["n_gen_gpes_matches"], ent_counter["n_gen_gpes"]),
        },
        "Entity ORG - recall": {
            "count": ent_counter["n_caption_orgs_matches"],
            "total": ent_counter["n_caption_orgs"],
            "percentage": safe_div(ent_counter["n_caption_orgs_matches"], ent_counter["n_caption_orgs"]),
        },
        "Entity ORG - precision": {
            "count": ent_counter["n_gen_orgs_matches"],
            "total": ent_counter["n_gen_orgs"],
            "percentage": safe_div(ent_counter["n_gen_orgs_matches"], ent_counter["n_gen_orgs"]),
        },
        "F1 (all entities)": {
            "percentage": f1_all,
        },
    }

    return entity_results


def eval_one_json(file_name: str, spacy_model: str = "en_core_web_lg"):
    ref_caption = []
    gen_caption = []

    with open(file_name, "r", encoding="utf-8") as f:
        data = json.load(f)

    for item in data:
        caption = item.get("caption", "")
        ref = item.get("ref_caption", "")

        if "</s>" in caption:
            caption = caption.split("</s>")[0]
        if "</s>" in ref:
            ref = ref.split("</s>")[0]

        ref_caption.append(process_string(ref))
        gen_caption.append(process_string(caption))

    results = evaluate_entity_by_gtent(ref_caption, gen_caption, spacy_model=spacy_model)
    return results


def main():
    parser = argparse.ArgumentParser(description="Entity-level evaluation based on spaCy NER")
    parser.add_argument(
        "--input_path",
        "-i",
        type=str,
        required=True,
        help="Path to a folder containing JSON result files",
    )
    parser.add_argument(
        "--spacy_model",
        "-m",
        type=str,
        default="en_core_web_lg",
        help="spaCy model name (e.g., en_core_web_lg)",
    )
    args = parser.parse_args()

    input_path = args.input_path
    if not os.path.isdir(input_path):
        raise ValueError(f"input_path must be a directory, got: {input_path}")

    json_files = sorted([f for f in os.listdir(input_path) if f.lower().endswith(".json")])
    for fname in json_files:
        fpath = os.path.join(input_path, fname)
        results = eval_one_json(fpath, spacy_model=args.spacy_model)
        print(fpath)
        print(results)


if __name__ == "__main__":
    main()