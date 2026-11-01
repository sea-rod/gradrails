import torch
import matplotlib.pyplot as plt
import datasets
import evaluate
import numpy as np
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    data,
)

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
OUTPUT_DIR = "./bert_output_mini"
ID2LABEL = {0: "False", 1: "True"}
LABEL2ID = {v: k for k, v in ID2LABEL.items()}
INPUT_DATA_PATH = "./data/input_data"


class BertClassifierPipeline:
    def __init__(
        self,
        model_name: str = MODEL_NAME,
        id2label: dict = ID2LABEL,
        label2id: dict = LABEL2ID,
        output_dir: str = OUTPUT_DIR,
        label_column: str = "label",
        feature_column: str = "prompt",
    ):
        self.model_name = model_name
        self.id2label = id2label
        self.label2id = label2id
        self.output_dir = output_dir
        self.label_column = label_column
        self.feature_column = feature_column

        self.tokenizer, self.model = self.load_model(self.model_name)
        self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

        self.accuracy_metric = evaluate.load("accuracy")
        self.f1_metric = evaluate.load("f1")
        self.recall_metric = evaluate.load("recall")
        self.precision_metric = evaluate.load("precision")

    def load_model(self, model_name):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=len(self.id2label),
            id2label=self.id2label,
            label2id=self.label2id,
        )
        model.to(self.device)

        return tokenizer, model

    def freeze_backbone(self):
        for name, param in self.model.named_parameters():
            if name.startswith("classifier"):
                param.requires_grad = True
            else:
                param.requires_grad = False
            print(f"Parameter: {name}, Requires Grad: {param.requires_grad}")

    def plot_training(self, trainer, save_path="training_curves.png"):
        history = trainer.state.log_history

        train_steps = [h["step"] for h in history if "loss" in h]
        train_loss = [h["loss"] for h in history if "loss" in h]

        eval_epochs = [h["epoch"] for h in history if "eval_loss" in h]
        eval_loss = [h["eval_loss"] for h in history if "eval_loss" in h]

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # Loss curves
        axes[0].plot(train_steps, train_loss, label="Train loss")
        axes[0].set_xlabel("Step")
        axes[0].set_ylabel("Loss")
        axes[0].set_title("Training loss")
        axes[0].legend()

        # Eval metrics per epoch
        axes[1].plot(eval_epochs, eval_loss, marker="o", label="Eval loss")
        for metric in ["accuracy", "f1", "recall", "precision"]:
            values = [h[f"eval_{metric}"] for h in history if f"eval_{metric}" in h]
            if values:
                axes[1].plot(eval_epochs, values, marker="o", label=metric)
        axes[1].set_xlabel("Epoch")
        axes[1].set_title("Evaluation metrics")
        axes[1].legend()

        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.show()

    def compute_metrics(self, eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)

        accuracy = self.accuracy_metric.compute(
            predictions=predictions, references=labels
        )
        f1 = self.f1_metric.compute(predictions=predictions, references=labels)
        recall = self.recall_metric.compute(predictions=predictions, references=labels)
        precision = self.precision_metric.compute(
            predictions=predictions, references=labels
        )
        return {
            "accuracy": accuracy["accuracy"],
            "f1": f1["f1"],
            "recall": recall["recall"],
            "precision": precision["precision"],
        }

    @staticmethod
    def load_data():
        data = datasets.load_dataset(INPUT_DATA_PATH, split="train")
        split_data = data.train_test_split(
            test_size=0.1, stratify_by_column="label", seed=42
        )
        return split_data["train"], split_data["test"]

    @staticmethod
    def normalize_data(data, prompt: str):
        data = list(data[prompt])
        return [" ".join(item.strip().lower().split()) for item in data]

    def validate_data(self, train_data, eval_data):
        train_data = self.normalize_data(train_data, self.feature_column)
        eval_data = self.normalize_data(eval_data, self.feature_column)

        U_train = set(train_data)
        U_eval = set(eval_data)

        leaked = U_eval & U_train
        if leaked:
            for sample in leaked[:10]:
                print(f"Leaked sample: {sample}\nTotal leaked samples: {len(leaked)}")
            raise ValueError(
                "The evaluation data contains samples that are also present in the training data."
            )

        duplicate_train = len(train_data) - len(U_train)
        duplicate_eval = len(eval_data) - len(U_eval)

        print(f"Duplicate samples in training data: {duplicate_train}")
        print(f"Duplicate samples in evaluation data: {duplicate_eval}")

    def format_data(self, data):
        tokenized = self.tokenizer(data[self.feature_column], truncation=True)
        data["input_ids"] = tokenized["input_ids"]
        data["attention_mask"] = tokenized["attention_mask"]
        data["token_type_ids"] = tokenized.get("token_type_ids", None)
        return data

    def train_model(self, data, eval_data):
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            hub_model_id="sea-rod/gaurdrails-miniLM-L6-V2",
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            learning_rate=1e-3,
            num_train_epochs=15,
            save_strategy="epoch",
            eval_strategy="epoch",
            logging_strategy="steps",
            logging_steps=10,
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            data_collator=self.data_collator,
            train_dataset=data,
            eval_dataset=eval_data,
            processing_class=self.tokenizer,
            compute_metrics=self.compute_metrics,
        )

        return trainer

    def run(self):
        self.freeze_backbone()

        train_data, eval_data = self.load_data()
        print(
            f"Loaded dataset with {len(train_data)} training samples and {len(eval_data)} evaluation samples."
        )
        self.validate_data(train_data, eval_data)

        train_data = train_data.map(self.format_data, batched=True)
        eval_data = eval_data.map(self.format_data, batched=True)
        trainer = self.train_model(train_data, eval_data)
        training_result = trainer.train()
        print("Training completed.")
        print("Training metrics:", training_result.metrics)
        self.plot_training(trainer)
        print("Evaluation metrics:", trainer.evaluate())

        trainer.save_model("./output/gaurdrails-miniLM-L6-V2")


def main():
    pipeline = BertClassifierPipeline(label_column="label", feature_column="text")
    pipeline.run()


if __name__ == "__main__":
    main()
