import json
import os
import sys

import evaluate
from datasets import load_dataset
from transformers import pipeline

MODEL_PATH = os.environ.get("MODEL_PATH", "output_model_mini")
DATA_PATH = os.environ.get("DATA_PATH", "./data/input_data")
MIN_ACCURACY = float(os.environ.get("MIN_ACCURACY",0.6))

ID2LABEL = {0: "False", 1: "True"}
LABEL2ID = {v: k for k, v in ID2LABEL.items()}


def compute_metrics(labels, predictions):
    names = ["accuracy", "f1", "recall", "precision"]
    return {
        n: evaluate.load(n).compute(predictions=predictions, references=labels)[n]
        for n in names
    }


def main():
    test_data = load_dataset(DATA_PATH, split="test",revision="dev")
    pipe = pipeline("text-classification", MODEL_PATH, tokenizer=MODEL_PATH,revision="dev")

    result = pipe(list(test_data["text"]), truncation=True)
    preds = [LABEL2ID[r["label"]] for r in result]

    metrics = compute_metrics(test_data["label"], preds)
    print(json.dumps(metrics, indent=2))

    if metrics["accuracy"] < MIN_ACCURACY:
        print(f"FAIL: accuracy {metrics['accuracy']:.4f} < {MIN_ACCURACY}")
        sys.exit(1)

    print(f"PASS: accuracy {metrics['accuracy']:.4f} >= {MIN_ACCURACY}")


if __name__ == "__main__":
    main()