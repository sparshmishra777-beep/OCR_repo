"""
infer_donut.py
---------------
Run a fine-tuned Donut model (produced by train_donut.py) on a new
receipt image and get back JSON.

Usage:
    python infer_donut.py --model_dir donut-receipts --image path/to/receipt.jpg
"""

import argparse
import json
import re

import torch
from PIL import Image
from transformers import DonutProcessor, VisionEncoderDecoderModel

TASK_START_TOKEN = "<s_receipt>"


def token2json(tokens: str) -> dict:
    """Inverse of train_donut.json2token: turn the model's XML-like
    output back into a Python dict."""
    tokens = tokens.replace(TASK_START_TOKEN, "").replace("</s_receipt>", "")
    tokens = re.sub(r"<s_(.*?)>", r'{"\1":', tokens)
    tokens = re.sub(r"</s_.*?>", "},", tokens)
    # This is a simplified reconstruction; for production use, prefer
    # donut's own `processor.token2json` utility if your Donut version
    # exposes it (transformers>=4.25 does, via DonutProcessor).
    return tokens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--image", required=True)
    args = ap.parse_args()

    processor = DonutProcessor.from_pretrained(args.model_dir)
    model = VisionEncoderDecoderModel.from_pretrained(args.model_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    image = Image.open(args.image).convert("RGB")
    pixel_values = processor(image, return_tensors="pt").pixel_values.to(device)

    decoder_input_ids = torch.tensor(
        [[model.config.decoder_start_token_id]]
    ).to(device)

    outputs = model.generate(
        pixel_values,
        decoder_input_ids=decoder_input_ids,
        max_length=768,
        early_stopping=True,
        pad_token_id=processor.tokenizer.pad_token_id,
        eos_token_id=processor.tokenizer.eos_token_id,
        use_cache=True,
        num_beams=1,
        bad_words_ids=[[processor.tokenizer.unk_token_id]],
        return_dict_in_generate=True,
    )

    sequence = processor.batch_decode(outputs.sequences)[0]
    sequence = sequence.replace(processor.tokenizer.eos_token, "")
    sequence = sequence.replace(processor.tokenizer.pad_token, "")

    # Preferred: use built-in token2json if available on your transformers version
    if hasattr(processor, "token2json"):
        result = processor.token2json(sequence)
    else:
        result = token2json(sequence)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
