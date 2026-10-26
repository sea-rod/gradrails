import datasets
import evaluate
import numpy as np
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
ID2LABEL = {0: "False", 1: "True"}
LABEL2ID = {v: k for k, v in ID2LABEL.items()}
OUTPUT_DIR = "./bert_output"


class BertClassifierPipeline:
    def __init__(
        self,
        model_name: str = MODEL_NAME,
        id2label: dict = None,
        output_dir: str = OUTPUT_DIR,
    ):
        self.model_name = model_name
        self.id2label = id2label if id2label is not None else ID2LABEL
        self.label2id = {v: k for k, v in self.id2label.items()}
        self.output_dir = output_dir

        self.tokenizer, self.model = self.load_model(self.model_name)
        self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

        self.accuracy_metric = evaluate.load("accuracy")
        self.f1_metric = evaluate.load("f1")
        self.recall_metric = evaluate.load("recall")
        self.precision_metric = evaluate.load("precision")

    def load_model(self, model_name):
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=len(self.id2label),
            id2label=self.id2label,
            label2id=self.label2id,
        )
        return tokenizer, model

    def freeze_backbone(self):
        for name, param in self.model.named_parameters():
            if name.startswith("classifier"):
                param.requires_grad = True
            else:
                param.requires_grad = False
            print(f"Parameter: {name}, Requires Grad: {param.requires_grad}")


    def compute_metrics(self, eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)

        accuracy = self.accuracy_metric.compute(
            predictions=predictions, references=labels
        )
        f1 = self.f1_metric.compute(
            predictions=predictions, references=labels, average="weighted"
        )
        recall = self.recall_metric.compute(
            predictions=predictions, references=labels, average="weighted"
        )
        precision = self.precision_metric.compute(
            predictions=predictions, references=labels, average="weighted"
        )
        return {
            "accuracy": accuracy["accuracy"],
            "f1": f1["f1"],
            "recall": recall["recall"],
            "precision": precision["precision"],
        }


    @staticmethod
    def load_data():
        return datasets.from_json("input/data.json")

    @staticmethod
    def normalize_data(data, prompt: str = "prompt"):
        data = list(data[prompt])
        return [" ".join(item.strip().lower().split()) for item in data]

    def validate_data(self, train_data, eval_data):
        train_data = self.normalize_data(train_data)
        eval_data = self.normalize_data(eval_data)

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

    def format_data(self, data, prompt: str = "prompt", labels: str = "label"):
        data[prompt] = data.map(lambda x: self.normalize_data(x, prompt), batched=True)
        data[labels] = [self.label2id[item] for item in data[labels]]
        tokenized = self.tokenizer(
            data, padding=True, truncation=True, return_tensors="pt"
        )
        data["input_ids"] = tokenized["input_ids"]
        data["attention_mask"] = tokenized["attention_mask"]
        data["token_type_ids"] = tokenized.get("token_type_ids", None)
        return data


    def train_model(self, data, eval_data):
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            learning_rate=2e-5,
            num_train_epochs=3,
            save_strategy="epoch",
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

        return trainer.train()

    def run(self):
        self.freeze_backbone()

        train_data, eval_data = self.load_data()
        print(
            f"Loaded dataset with {len(train_data)} training samples and {len(eval_data)} evaluation samples."
        )
        self.validate_data(train_data, eval_data)

        train_data = train_data.map(lambda x: self.format_data(x))
        eval_data = eval_data.map(lambda x: self.format_data(x))
        training_result = self.train_model(train_data, eval_data)
        print("Training completed.")
        print("Training metrics:", training_result.metrics)


def main():
    pipeline = BertClassifierPipeline()
    pipeline.run()


if __name__ == "__main__":
    main()
