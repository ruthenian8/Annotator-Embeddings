#!/usr/bin/env python
# all rights belong to chatgpt
import os
import json
import argparse
from collections import defaultdict
from copy import deepcopy

import pandas as pd

def main(args):
    # prepare output directories
    processed_dir = f"Annotator-Embeddings/src/example-data/{args.dataset_name}-processed/"
    train_dir = f"{processed_dir}annotation_split_train"
    test_dir  = f"{processed_dir}annotation_split_test"
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir,  exist_ok=True)

    hf_dir = f"Annotator-Embeddings/huggingface-data/{args.dataset_name}-ann"
    os.makedirs(hf_dir, exist_ok=True)

    df = pd.read_csv(args.csv)

    # 1) Extract all annotator IDs
    annotators = sorted(df['annotator'].astype(str).unique())
    annotator_ids = {aid: None for aid in annotators}
    with open(os.path.join(processed_dir, args.annotator_id_path), 'w') as f:
        json.dump(annotator_ids, f, indent=2)
    print(f"Wrote {len(annotators)} annotator IDs to {args.annotator_id_path}")

    # 2) Extract all label values
    labels = sorted(df['label'].astype(str).unique())
    with open(os.path.join(processed_dir, args.annotation_label_path), 'w') as f:
        json.dump(labels, f, indent=2)
    print(f"Wrote {len(labels)} labels to {args.annotation_label_path}")

    # 3) Build all_annotations (with anns_except_current_one)
    grouped = df.groupby('pair_id')
    all_annotations = []
    next_id = 0

    for pair_id, group in grouped:
        all_labels = group['label'].astype(str).tolist()
        for _, row in group.iterrows():
            ann = {
                "id": next_id,
                "sentence": row['prep_parent_text'] + " " + row['prep_text'],
                args.task_name: str(row['label']),
                "respondent_id": str(row['annotator']),
                "anns_except_current_one": [lab for lab in all_labels if lab != str(row['label'])],
                # use the original split column:
                "split": row['split'].strip().lower()
            }
            all_annotations.append(ann)
            next_id += 1

    print(f"Built {next_id} total annotations from {len(grouped)} pairs")

    # 4) Write out JSON + JSONL and build manifests
    train_manifest = []
    test_manifest  = []

    train_jsonl = open(os.path.join(hf_dir, "train.jsonl"), "w")
    test_jsonl  = open(os.path.join(hf_dir, "test.jsonl"),  "w")

    for ann in all_annotations:
        if ann["split"] == "train":
            out_dir = train_dir
            manifest = train_manifest
            jsonl_f = train_jsonl
            prefix = "train"
        elif ann["split"] == "test":
            out_dir = test_dir
            manifest = test_manifest
            jsonl_f = test_jsonl
            prefix = "test"
        else:
            # skip any unrecognized split
            continue

        fname = f"{prefix}_{ann['id']}.json"
        path = os.path.join(out_dir, fname)
        with open(path, 'w') as f:
            # drop the split field in the final JSON
            ann_to_dump = {k:v for k,v in ann.items() if k != "split"}
            json.dump(ann_to_dump, f, indent=4)

        # record in manifest (absolute path)
        manifest.append({
            "id": ann["id"],
            "path": os.path.abspath(path)
        })

        # also write raw to JSONL (without split)
        jsonl_f.write(json.dumps(ann_to_dump) + "\n")

    train_jsonl.close()
    test_jsonl.close()

    # 5) Write split manifests
    with open(os.path.join(processed_dir, "annotation_split_train.json"), 'w') as f:
        json.dump(train_manifest, f, indent=4)
    with open(os.path.join(processed_dir, "annotation_split_test.json"), 'w') as f:
        json.dump(test_manifest, f, indent=4)

    print(f"Done.  Train examples: {len(train_manifest)}, Test examples: {len(test_manifest)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess CSV→Annotator‐Embeddings JSON format (use original splits)"
    )
    parser.add_argument(
        "--csv", type=str, required=True,
        help="path to all_data.csv"
    )
    parser.add_argument(
        "--dataset_name", type=str, required=True,
        help="a short name (e.g. `custom`) to prefix output dirs"
    )
    parser.add_argument(
        "--task_name", type=str, required=True,
        help="the column in your CSV that holds the label (e.g. `label`)"
    )
    parser.add_argument(
        "--annotator_id_path", type=str, default="annotator_ids.json",
        help="where to write the JSON object of annotator IDs"
    )
    parser.add_argument(
        "--annotation_label_path", type=str, default="annotation_labels.json",
        help="where to write the JSON list of label values"
    )
    args = parser.parse_args()
    main(args)
