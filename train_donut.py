"""
train_donut.py
---------------
Fine-tunes a Donut model (naver-clova-ix/donut-base) to read a receipt
image and directly output a JSON string, with no separate OCR step.
This is the standard approach for "image -> structured JSON" document
understanding tasks (it's what SROIE-style receipt datasets are normally
solved with) and will outperform the rule-based OCR+regex pipeline
(parser.py) once you have labeled data, especially for messy/rotated
photos and stylized merchant logos.

REQUIREMENTS (not installed in this sandbox — run this on a machine/
Colab notebook with a GPU):
    pip install transformers datasets torch torchvision pillow sentencepiece accelerate

EXPECTED DATASET FORMAT
------------------------
Give me (or point this script at) a folder like:

    dataset/
      images/
        receipt_0001.png
        receipt_0002.png
        ...
      metadata.jsonl        <- one line per image

Each line of metadata.jsonl should look like:
    {"file_name": "receipt_0001.png", "ground_truth": {
        "merchant_name": "WAL-MART",
        "date": "08/20/10",
        "time": "13:12:01",
        "line_items": [
            {"description": "BANANAS", "amount": 0.20},
            {"description": "FRAP", "amount": 5.48}
        ],
        "subtotal": 5.11,
        "tax": null,
        "total": 5.11,
        "cash_tendered": 11.00,
        "change_due": 5.89,
        "payment_method": "cash",
        "currency": "USD"
    }}

This is exactly the schema parser.py already outputs, so you can
bootstrap labels by running run_pipeline.py and hand-correcting its
output rather than labeling from scratch.

USAGE
-----
    python train_donut.py --data_dir dataset --output_dir donut-receipts --epochs 5
"""

import argparse
import json
import os

from datasets import load_dataset
from PIL import Image
import torch
from transformers import (
    DonutProcessor,
    VisionEncoderDecoderModel,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

TASK_START_TOKEN = "<s_receipt>"
TASK_END_TOKEN = "</s_receipt>"
MAX_LENGTH = 768


def json2token(obj):
    """Flatten a JSON dict into Donut's XML-like token sequence."""
    if isinstance(obj, dict):
        tokens = ""
        for k, v in obj.items():
            tokens += f"<s_{k}>" + json2token(v) + f"</s_{k}>"
        return tokens
    elif isinstance(obj, list):
        return "".join(json2token(item) for item in obj)
    else:
        return str(obj) if obj is not None else ""


def build_dataset(data_dir, processor):
    ds = load_dataset(
        "imagefolder",
        data_dir=os.path.join(data_dir, "images"),
        split="train",
    )
    # attach ground truth from metadata.jsonl (imagefolder auto-loads a
    # "metadata.jsonl" sitting next to the images if present)

    def preprocess(example):
        image = example["image"].convert("RGB")
        pixel_values = processor(image, return_tensors="pt").pixel_values.squeeze()

        gt = example["ground_truth"]
        if isinstance(gt, str):
            gt = json.loads(gt)
        target_sequence = TASK_START_TOKEN + json2token(gt) + TASK_END_TOKEN

        labels = processor.tokenizer(
            target_sequence,
            add_special_tokens=False,
            max_length=MAX_LENGTH,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).input_ids.squeeze()
        labels[labels == processor.tokenizer.pad_token_id] = -100

        return {"pixel_values": pixel_values, "labels": labels}

    return ds.map(preprocess, remove_columns=ds.column_names)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, help="Folder with images/ and metadata.jsonl")
    ap.add_argument("--output_dir", default="donut-receipts")
    ap.add_argument("--base_model", default="naver-clova-ix/donut-base")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--lr", type=float, default=3e-5)
    args = ap.parse_args()

    processor = DonutProcessor.from_pretrained(args.base_model)
    model = VisionEncoderDecoderModel.from_pretrained(args.base_model)

    processor.tokenizer.add_special_tokens(
        {"additional_special_tokens": [TASK_START_TOKEN, TASK_END_TOKEN]}
    )
    model.decoder.resize_token_embeddings(len(processor.tokenizer))
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.decoder_start_token_id = processor.tokenizer.convert_tokens_to_ids(
        TASK_START_TOKEN
    )

    train_dataset = build_dataset(args.data_dir, processor)

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        fp16=torch.cuda.is_available(),
        logging_steps=10,
        save_strategy="epoch",
        predict_with_generate=True,
        remove_unused_columns=False,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)
    print(f"Model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
