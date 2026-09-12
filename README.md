# Receipt → JSON pipeline

Two tiers, so you get useful output today and a path to something much
more accurate once you share a labeled dataset.

## 1. Working now: OCR + rule-based parser (no training required)

- `parser.py` — turns raw OCR text into a structured JSON record
  (merchant name, date, time, line items, subtotal, tax, total, cash
  tendered, change due, payment method, currency).
- `run_pipeline.py` — runs Tesseract OCR (with grayscale + autocontrast
  preprocessing, which matters a lot for photographed receipts) over a
  folder of images and writes one JSON record per receipt.

Run it:
```bash
python run_pipeline.py            # reads images/*.png, writes output/all_receipts.json
```

I ran this on the 62 receipts from your PDF — see `output/all_receipts.json`.
It gets totals/tax/line-items right on clean, front-facing receipts, but
struggles with:
- stylized logos (e.g. "WAL★MART", "Trader Joe's" script font) for merchant
  name detection
- receipts photographed at an angle, in a hand, or on a patterned
  background
- non-USD currency symbols it hasn't seen (e.g. "RM" for Malaysian Ringgit
  gets picked up as text, not tagged as a currency correctly in all cases)

This is a reasonable baseline / fallback, and it needs **zero training
data** — but for meaningfully better accuracy on messy real-world photos,
you want option 2.

## 2. Trainable model: Donut (image → JSON directly, no OCR step)

`train_donut.py` fine-tunes a `donut-base` vision-encoder-decoder model
that reads the receipt image directly and generates the JSON as output —
no separate OCR/regex step, so it's far more robust to skew, stylized
fonts, and background clutter. This is the standard approach used for
datasets like SROIE (which is what your `all_receipts.pdf` images look
like they're drawn from).

### What I need from you
A folder like:
```
dataset/
  images/
    receipt_0001.png
    receipt_0002.png
    ...
  metadata.jsonl
```
where `metadata.jsonl` has one line per image:
```json
{"file_name": "receipt_0001.png", "ground_truth": {"merchant_name": "WAL-MART", "date": "08/20/10", "total": 5.11, ...}}
```
Same field names as `parser.py`'s output — so the fastest way to build
this is: run `run_pipeline.py` on your images, then hand-correct the
JSON it produces rather than labeling from scratch.

### Training (needs a GPU — Colab, or your own machine; this sandbox has none)
```bash
pip install transformers datasets torch torchvision pillow sentencepiece accelerate
python train_donut.py --data_dir dataset --output_dir donut-receipts --epochs 5
```

### Inference
```bash
python infer_donut.py --model_dir donut-receipts --image new_receipt.jpg
```

## Files
| File | Purpose |
|---|---|
| `parser.py` | rule-based OCR-text → JSON extractor |
| `run_pipeline.py` | OCR + parser, run over a folder of images |
| `train_donut.py` | fine-tune Donut on your labeled dataset |
| `infer_donut.py` | run the fine-tuned Donut model on a new receipt |
| `images/` | the 62 receipt images extracted from your PDF |
| `output/all_receipts.json` | baseline pipeline's output on those 62 receipts |

## Next step
Send over your dataset (or point me at where it lives) and I'll kick off
training and validate accuracy on a held-out split.
