import numpy as np

from pathlib import Path

from transformers import T5Tokenizer, T5ForConditionalGeneration, DataCollatorForSeq2Seq, Seq2SeqTrainer, Seq2SeqTrainingArguments
from datasets import Dataset, ClassLabel

import evaluate

from tqdm.auto import tqdm
from typing import Tuple

# pylint: disable-next=relative-beyond-top-level
from ..utils import evaluate_model, get_device

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

    # add tokens for role combinations
    # num_added_tokens += tokenizer.add_tokens(
    #     [f"{l}," for l in label_list if l != "0"])

    print(f"### {num_added_tokens} tokens have been added to the tokenizer")

    model.resize_token_embeddings(len(tokenizer))

    tokenized_data, dataset = _tokenize_data(dataset, tokenizer, label_list)

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    training_args = Seq2SeqTrainingArguments(
        output_dir=MODEL_FOLDER / "srl-generator" / "checkpoints",
        overwrite_output_dir=True,
        evaluation_strategy="epoch",
        save_strategy="no",
        learning_rate=1e-4,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
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

        decoded_preds = [pred.strip().replace(", ", ",")
                         for pred in decoded_preds]
        decoded_labels = [label.strip().replace(", ", ",")
                          for label in decoded_labels]

        bleu = metric.compute(predictions=decoded_preds,
                              references=decoded_labels)

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


def evaluate_generator(dataset: Dataset, model_name: str):

    label_feat = dataset["train"].features["verbnet"].feature.feature

    # uncomment for local testing
    dataset["train"] = dataset["train"].select(range(64))
    dataset["test"] = dataset["test"].select(range(64))

    model = T5ForConditionalGeneration.from_pretrained(
        MODEL_FOLDER / "srl-generator" / model_name)

    tokenizer = T5Tokenizer.from_pretrained(
        "t5-base", model_max_length=model.config.n_positions)

    num_added_tokens = tokenizer.add_tokens(label_feat.names)

    # add tokens for role combinations
    # num_added_tokens += tokenizer.add_tokens(
    #   [f"{l}," for l in label_feat.names if l != "0"])

    print(f"### {num_added_tokens} tokens have been added to the tokenizer")

    model.resize_token_embeddings(len(tokenizer))

    tokenized_data, dataset = _tokenize_data(dataset, tokenizer, label_feat.names)

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    device = get_device()
    print(f"### evaluating model on {device} ###")
    model.to(device)

    model.eval()
    model.zero_grad()

    input_texts = []
    labels = []
    preds = []
    gen_length = []

    progress_bar = tqdm(range(len(tokenized_data["test"])))

    for example in tokenized_data["test"]:

        label = example.pop("labels")
        decoded_labels = post_process_gen_outputs(label, tokenizer, label_feat)
        labels.append(decoded_labels)

        input_text = example.pop("input_text")
        input_texts.append(input_text)

        inputs = data_collator([example])

        inputs = {k: v.to(device) for k, v in inputs.items()}

        # we want to generate at least one new token per input token
        min_length = len(input_text)
        max_length = 3 * len(input_text)

        outputs = model.generate(
            **inputs, min_new_tokens=min_length, max_new_tokens=max_length, num_beams=4, early_stopping=True)

        decoded_pred = post_process_gen_outputs(
            outputs[0], tokenizer, label_feat)

        # special handling to avoid length mismatches between input and predictions
        if len(decoded_pred) > len(input_text):
            decoded_pred = decoded_pred[:len(input_text)]
            gen_length.append("long")
        elif len(decoded_pred) < len(input_text):
            missing = len(input_text) - len(decoded_pred)

            for i in range(missing):
                decoded_pred.append([0])
            gen_length.append("short")
        else:
            gen_length.append("equal")
        preds.append(decoded_pred)

        progress_bar.update(1)

    progress_bar.close()

    evaluate_model(dataset, input_texts, labels, preds, gen_length)


def post_process_gen_outputs(out: list, tokenizer: T5Tokenizer, class_label: ClassLabel) -> list:

    decoded = tokenizer.decode(out, skip_special_tokens=True)
    decoded = decoded.strip().replace(", ", ",").split(" ")

    decoded_class = []
    for tok in decoded:
        tok = tok.split(",")
        tok = class_label.str2int(tok)
        decoded_class.append(tok)

    return decoded_class


def _tokenize_data(ds: Dataset, tokenizer: T5Tokenizer, label_list: list) -> Tuple[Dataset, Dataset]:

    remove_examples = set()
    def tokenize(examples):

        inputs = [" ".join(example) for example in examples["tok"]]
        tokens = tokenizer(inputs, truncation=True,
                           return_overflowing_tokens=True)

        rm_ids = set()
        for idx, trunc_len in enumerate(tokens["num_truncated_tokens"]):
            if trunc_len > 0:
                rm_ids.add(idx)

        labels = []
        for sen_roles in examples["verbnet"]:
            label = []

            for roles in sen_roles:
                role_str = ",".join([label_list[r] for r in roles])
                label.append(role_str)

            labels.append(" ".join(label))

        tok_labels = tokenizer(labels, truncation=True,
                               return_overflowing_tokens=True)

        for idx, trunc_len in enumerate(tok_labels["num_truncated_tokens"]):
            if trunc_len > 0:
                rm_ids.add(idx)

        tokens["labels"] = tok_labels["input_ids"]
        tokens["input_text"] = examples["tok"]

        for idx in rm_ids:
            remove_examples.add(examples["id"][idx])
            for k, v in tokens.items():
                v.pop(idx)

        del tokens["num_truncated_tokens"]
        del tokens["overflowing_tokens"]

        return tokens

    tokenized_dataset = ds.map(
        tokenize, batched=True, remove_columns=ds["train"].column_names)

    tokenized_dataset.set_format(
        "torch", columns=["input_ids", "attention_mask"], output_all_columns=True)

    ds = ds.filter(lambda e: e["id"] not in remove_examples)

    return tokenized_dataset, ds
