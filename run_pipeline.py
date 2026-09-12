import os
import json
import glob
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

from PIL import Image, ImageOps
from parser import parse_receipt_text

IMG_DIR = "images"
OUT_DIR = "output"
os.makedirs(OUT_DIR, exist_ok=True)


def preprocess(img: Image.Image) -> Image.Image:
    """Grayscale + autocontrast tends to work well for photographed
    (non-flatbed-scanned) receipts with background clutter."""
    img = img.convert("L")
    img = ImageOps.autocontrast(img)
    return img


files = sorted(glob.glob(os.path.join(IMG_DIR, "*.jpg")))
print(f"Found {len(files)} images")

all_results = []
for i, path in enumerate(files):
    try:
        img = Image.open(path)
        img = preprocess(img)
        text = pytesseract.image_to_string(img, config="--psm 6")
        record = parse_receipt_text(text, source_file=os.path.basename(path))
        all_results.append(record)
    except Exception as e:
        print(f"Error on {path}: {e}")

    if (i + 1) % 10 == 0:
        print(f"Processed {i+1}/{len(files)}")

with open(os.path.join(OUT_DIR, "all_receipts.json"), "w") as f:
    json.dump(all_results, f, indent=2)

print(f"Done. Wrote {len(all_results)} records to {OUT_DIR}/all_receipts.json")
