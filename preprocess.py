#!/usr/bin/env python
# all rights belong to chatgpt
import os
import json
import argparse
from collections import defaultdict
from copy import deepcopy

import pandas as pd
from sklearn.model_selection import train_test_split

def main(args):
    # prepare output directories
    processed_dir = f"Annotator-Embeddings/src/example-data/{args.dataset_name}-processed/"
    train_dir = f"{processed_dir}annotation_split_train"
    test_dir  = f"{processed_dir}annotation_split_test"
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir,  exist_ok=True)

    os.makedirs(f"Annotator-Embeddings/huggingface-data/{args.dataset_name}-ann", exist_ok=True)

    df = pd.read_csv(args.csv)

    # 1) Extract all annotator IDs
    annotators = sorted(df['annotator'].astype(str).unique())
    annotator_ids = {aid: None for aid in annotators}
    with open(processed_dir + args.annotator_id_path, 'w+') as f:
        json.dump(annotator_ids, f, indent=2)
    print(f"Wrote {len(annotators)} annotator IDs to {args.annotator_id_path}")

    # 2) Extract all label values
    labels = sorted(df['label'].astype(str).unique())
    with open(processed_dir + args.annotation_label_path, 'w') as f:
        json.dump(labels, f, indent=2)
    print(f"Wrote {len(labels)} labels to {args.annotation_label_path}")

    # 3) Build annotator_data: mapping from annotator -> list of annotations
    #    We also need to keep track of all labels on each pair so we can compute anns_except_current_one.
    grouped = df.groupby('pair_id')
    annotator_data = defaultdict(list)
    next_id = 0

    for pair_id, group in grouped:
        # list of all labels on this pair
        all_labels = group['label'].astype(str).tolist()
        for _, row in group.iterrows():
            ann = {
                # a unique integer ID for this annotation
                "id": next_id,
                # the text that goes into the model
                "sentence": row['prep_parent_text'] + " " + row['prep_text'],
                # your task label key
                args.task_name: str(row['label']),
                # the annotator
                "respondent_id": str(row['annotator'])
            }
            # everyone's labels except this one
            others = deepcopy(all_labels)
            others.remove(str(row['label']))
            ann["anns_except_current_one"] = others

            annotator_data[str(row['annotator'])].append(ann)
            next_id += 1

    print(f"Built annotator_data for {len(annotator_data)} annotators, {next_id} total examples")

    train_jsonl = open(f"Annotator-Embeddings/huggingface-data/{args.dataset_name}-ann/train.jsonl", "w")
    test_jsonl  = open(f"Annotator-Embeddings/huggingface-data/{args.dataset_name}-ann/test.jsonl",  "w")

    train_manifest = []
    test_manifest  = []

    # 4) For each annotator, split their annotations into train/test
    for annotator, anns in annotator_data.items():
        if len(anns) > 1:
            train_anns, test_anns = train_test_split(
                anns, test_size=0.3, random_state=42)
        else:
            train_anns, test_anns = anns, []

        # write out train examples
        for ann in train_anns:
            path = os.path.join(train_dir, f"train_{ann['id']}.json")
            with open(path, 'w') as f:
                json.dump(ann, f, indent=4)
            train_manifest.append({"id": ann["id"], "path": os.path.abspath(path)})
            train_jsonl.write(json.dumps(ann) + "\n")

        # write out test examples
        for ann in test_anns:
            path = os.path.join(test_dir, f"test_{ann['id']}.json")
            with open(path, 'w') as f:
                json.dump(ann, f, indent=4)
            test_manifest.append({"id": ann["id"], "path": os.path.abspath(path)})
            test_jsonl.write(json.dumps(ann) + "\n")

    train_jsonl.close()
    test_jsonl.close()

    # 5) Write split manifests
    with open(f"{processed_dir}annotation_split_train.json", 'w') as f:
        json.dump(train_manifest, f, indent=4)
    with open(f"{processed_dir}annotation_split_test.json", 'w') as f:
        json.dump(test_manifest, f, indent=4)

    print(f"Train/test split written. Train: {len(train_manifest)}, Test: {len(test_manifest)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess CSV→Annotator‐Embeddings JSON format"
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


