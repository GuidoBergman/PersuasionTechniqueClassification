import os

from transformers import RobertaTokenizerFast, RobertaForTokenClassification, DataCollatorForTokenClassification, TrainingArguments, Trainer

from pathlib import Path
from datasets import Dataset

import evaluate
import numpy as np
import pprint

MODEL_FOLDER = Path(__file__).parent.parent.parent.resolve() / "models"


def train_classifier(dataset: Dataset):

    label_list = dataset["train"].features["verbnet"].feature.feature.names

    model = RobertaForTokenClassification.from_pretrained(
        "roberta-base", num_labels=len(label_list))

    tokenizer = RobertaTokenizerFast.from_pretrained(
        "roberta-base", add_prefix_space=True)

    def _compute_metrics(p):
        logits, labels = p

        pred = np.argmax(logits, axis=-1)

        str_pred = [
            [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(pred, labels)
        ]
        str_labels = [
            [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(pred, labels)
        ]

        results = eval_metric.compute(
            predictions=str_pred, references=str_labels, zero_division=0)
        return {
            "macro_avg": results["macro avg"],
            "weighted_avg": results["weighted avg"],
            "accuracy": results["accuracy"]
        }

    tokenized_data = _tokenize_data(dataset, tokenizer)

    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    eval_metric = evaluate.load("poseval")

    training_args = TrainingArguments(
        output_dir=MODEL_FOLDER / "srl-classifier" / "checkpoints",
        overwrite_output_dir=True,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        num_train_epochs=5,
        label_smoothing_factor=0.05,
        weight_decay=0.01,
        optim="adamw_torch"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_data["train"],
        eval_dataset=tokenized_data["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=_compute_metrics
    )

    trainer.train()

    trainer.save_model(output_dir=MODEL_FOLDER / "srl-classifier" / "model")


def evaluate_classifier(dataset: Dataset):

    label_list = dataset["train"].features["verbnet"].feature.feature.names

    model = RobertaForTokenClassification.from_pretrained(
        MODEL_FOLDER / "srl-classifier" / "model",  num_labels=len(label_list))

    tokenizer = RobertaTokenizerFast.from_pretrained(
        "roberta-base", add_prefix_space=True)

    tokenized_data = _tokenize_data(dataset, tokenizer)

    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    eval_metric = evaluate.load("poseval")

    def _compute_metrics(p):
        logits, labels = p

        pred = np.argmax(logits, axis=-1)

        str_pred = [
            [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(pred, labels)
        ]
        str_labels = [
            [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(pred, labels)
        ]

        return eval_metric.compute(
            predictions=str_pred, references=str_labels, zero_division=0)

    training_args = TrainingArguments(
        output_dir=MODEL_FOLDER / "srl-classifier" / "checkpoints",
        overwrite_output_dir=True,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        num_train_epochs=5,
        label_smoothing_factor=0.05,
        weight_decay=0.01,
        optim="adamw_torch"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_data["train"],
        eval_dataset=tokenized_data["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=_compute_metrics
    )

    metrics = trainer.evaluate()

    pprint.pprint(metrics)


def _tokenize_data(dataset: Dataset, tokenizer: RobertaTokenizerFast, label_all_tokens: bool = True) -> Dataset:

    def tokenize_and_align(examples):
        tokens = tokenizer(examples["tok"], is_split_into_words=True)

        labels = []
        for i, label in enumerate(examples["verbnet"]):
            word_ids = tokens.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:

                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:
                    # only set a single label for now
                    label_ids.append(label[word_idx][0])
                else:
                    label_ids.append(label[word_idx][0]
                                     if label_all_tokens else -100)

                previous_word_idx = word_idx

            labels.append(label_ids)

        tokens["labels"] = labels

        return tokens

    tokenized_dataset = dataset.map(tokenize_and_align, batched=True, num_proc=os.cpu_count(
    ), remove_columns=dataset["train"].column_names)

    return tokenized_dataset
