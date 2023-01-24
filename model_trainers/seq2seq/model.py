import os

import numpy as np

from pathlib import Path

from transformers import T5Tokenizer, T5ForConditionalGeneration, DataCollatorForSeq2Seq, Seq2SeqTrainer, Seq2SeqTrainingArguments

from datasets import Dataset

import evaluate

#from sklearn.metrics import classification_report

MODEL_FOLDER = Path(__file__).parent.parent.parent.resolve() / "models"


def train_generator(dataset: Dataset, model_name: str, num_epochs: int = 3):

    label_list = dataset["train"].features["verbnet"].feature.feature.names

    # uncomment for local testing
    # dataset["train"] = dataset["train"].select(range(64))
    # dataset["test"] = dataset["test"].select(range(64))

    model = T5ForConditionalGeneration.from_pretrained("t5-base")

    tokenizer = T5Tokenizer.from_pretrained(
        "t5-base", model_max_length=model.config.n_positions)

    num_added_tokens = tokenizer.add_tokens(label_list)

    print(f"### {num_added_tokens} tokens have been added to the tokenizer")

    model.resize_token_embeddings(len(tokenizer))

    tokenized_data = _tokenize_data(dataset, tokenizer, label_list)

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    training_args = Seq2SeqTrainingArguments(
        output_dir=MODEL_FOLDER / "srl-generator" / "checkpoints",
        overwrite_output_dir=True,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=1e-4,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        num_train_epochs=num_epochs,
        weight_decay=0.01,
        predict_with_generate=True,
        optim="adafactor"
    )

    metric = evaluate.load("bleu")

    def compute_metrics(p):
        preds, labels = p

        if isinstance(preds, tuple):
            preds = preds[0]

        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(
            labels, skip_special_tokens=True)

        decoded_preds = [pred.strip().replace(", ", ",") for pred in decoded_preds]
        decoded_labels = [label.strip().replace(", ", ",") for label in decoded_labels]

        bleu = metric.compute(predictions=decoded_preds, references=decoded_labels)

        len_mismatch = 0
        for pred, label in zip(decoded_preds, decoded_labels):
            pred_list = pred.split(" ")
            label_list = label.split(" ")
            if len(pred_list) != len(label_list):
                len_mismatch += 1

        return {
            "len_mismatch": len_mismatch / len(preds),
            "bleu": bleu["bleu"],
            "bleu_length_ratio": bleu["length_ratio"]
        }

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_data["train"],
        eval_dataset=tokenized_data["test"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    trainer.train()

    trainer.save_model(output_dir=MODEL_FOLDER / "srl-generator" / model_name)


def _tokenize_data(ds: Dataset, tokenizer: T5Tokenizer, label_list: list) -> Dataset:

    def tokenize(examples):

        inputs = [" ".join(example) for example in examples["tok"]]
        tokens = tokenizer(inputs)

        labels = []
        for sen_roles in examples["verbnet"]:
            label = []

            for roles in sen_roles:
                role_str = ",".join([label_list[r] for r in roles])
                label.append(role_str)

            labels.append(" ".join(label))

        tok_labels = tokenizer(labels)

        tokens["labels"] = tok_labels["input_ids"]

        return tokens

    tokenized_dataset = ds.map(
        tokenize, batched=True, remove_columns=ds["train"].column_names, num_proc=os.cpu_count())

    return tokenized_dataset
