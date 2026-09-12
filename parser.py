"""
receipt_parser.py
------------------
Rule-based field extractor that turns raw OCR text (from Tesseract, or any
other OCR engine) into a structured JSON record for a receipt.

This is the "baseline" / zero-training model in the pipeline. It works on
any receipt right now, with no dataset required. It's also the reference
implementation for the fields that the trainable model (train_donut.py)
will be asked to predict, so the JSON schema stays consistent whichever
model produced it.

Usage:
    from parser import parse_receipt_text
    result = parse_receipt_text(raw_ocr_text)
"""

import re
from datetime import datetime


MONEY_RE = re.compile(r"(-?\$?(?:RM)?\s?-?\d[\d,]*\.\d{2})")
DATE_PATTERNS = [
    (r"\b(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})\b", "%m/%d/%y"),   # 08/20/10, 06-28-2014
    (r"\b(\d{4})[/-](\d{1,2})[/-](\d{1,2})\b", "%Y-%m-%d"),      # 2014-06-28
]
TIME_RE = re.compile(r"\b(\d{1,2}):(\d{2})(?::(\d{2}))?\s?(AM|PM|am|pm)?\b")

TOTAL_KEYWORDS = ["total", "grand total", "amount due", "total sales", "tot'l", "tot' l"]
SUBTOTAL_KEYWORDS = ["subtotal", "sub total", "sub-total"]
TAX_KEYWORDS = ["tax", "gst", "vat"]
CASH_KEYWORDS = ["cash tend", "cash tendered", "cash"]
CHANGE_KEYWORDS = ["change due", "change"]
CARD_KEYWORDS = ["visa", "debit", "mastercard", "eft debit", "credit"]


def _clean_money(s: str) -> float:
    s = s.replace("RM", "").replace("$", "").replace(",", "").strip()
    try:
        return round(float(s), 2)
    except ValueError:
        return None


def _find_amount_on_line(line: str):
    matches = MONEY_RE.findall(line)
    if matches:
        return _clean_money(matches[-1])
    return None


def _guess_date(text: str):
    for pattern, _ in DATE_PATTERNS:
        m = re.search(pattern, text)
        if m:
            return m.group(0)
    return None


def _guess_time(text: str):
    m = TIME_RE.search(text)
    if m:
        return m.group(0)
    return None


def _guess_merchant(lines):
    """Merchant name is almost always one of the first non-empty lines,
    usually the longest / most 'title-like' line before numbers show up."""
    candidates = []
    for line in lines[:6]:
        clean = line.strip()
        if not clean:
            continue
        if MONEY_RE.search(clean):
            continue
        if re.search(r"\d{3,}", clean):  # phone numbers, addresses etc, deprioritize
            continue
        candidates.append(clean)
    if candidates:
        # Prefer all-caps / title-like short lines (store names are usually short)
        candidates.sort(key=lambda c: len(c))
        return candidates[0]
    return lines[0].strip() if lines else None


def _extract_line_items(lines):
    """
    Heuristic line-item extractor: a line that ends in a money amount and
    isn't a known summary keyword (total/tax/subtotal/etc) is treated as
    a purchased item.
    """
    items = []
    skip_keywords = (
        TOTAL_KEYWORDS + SUBTOTAL_KEYWORDS + TAX_KEYWORDS +
        CASH_KEYWORDS + CHANGE_KEYWORDS + CARD_KEYWORDS +
        ["items sold", "approval", "account", "ref #", "network id",
         "terminal", "change due", "tender", "auth", "trace"]
    )
    for line in lines:
        low = line.lower()
        if any(k in low for k in skip_keywords):
            continue
        amt = _find_amount_on_line(line)
        if amt is None:
            continue
        # strip the trailing amount + tax flag letters (N/X/F/O/T/etc) to get description
        desc = MONEY_RE.sub("", line).strip()
        desc = re.sub(r"\s[A-Z]{1,2}\*?$", "", desc).strip(" -:")
        if not desc:
            continue
        items.append({"description": desc, "amount": amt})
    return items


def parse_receipt_text(raw_text: str, source_file: str = None) -> dict:
    lines = [l for l in raw_text.splitlines() if l.strip()]
    full_text = "\n".join(lines)

    result = {
        "source_file": source_file,
        "merchant_name": _guess_merchant(lines),
        "date": _guess_date(full_text),
        "time": _guess_time(full_text),
        "line_items": _extract_line_items(lines),
        "subtotal": None,
        "tax": None,
        "total": None,
        "cash_tendered": None,
        "change_due": None,
        "payment_method": None,
        "currency": "USD" if "RM" not in full_text else "MYR",
        "raw_text": raw_text,
    }

    for line in lines:
        low = line.lower()
        if any(k in low for k in SUBTOTAL_KEYWORDS) and result["subtotal"] is None:
            result["subtotal"] = _find_amount_on_line(line)
        elif any(k in low for k in TAX_KEYWORDS) and result["tax"] is None:
            amt = _find_amount_on_line(line)
            if amt is not None:
                result["tax"] = (result["tax"] or 0) + amt
        elif any(k in low for k in CASH_KEYWORDS) and result["cash_tendered"] is None:
            result["cash_tendered"] = _find_amount_on_line(line)
        elif any(k in low for k in CHANGE_KEYWORDS) and result["change_due"] is None:
            result["change_due"] = _find_amount_on_line(line)

        if any(k in low for k in TOTAL_KEYWORDS) and "subtotal" not in low:
            amt = _find_amount_on_line(line)
            if amt is not None:
                result["total"] = amt

        if result["payment_method"] is None:
            if any(k in low for k in CARD_KEYWORDS):
                result["payment_method"] = "card"
            elif "cash" in low:
                result["payment_method"] = "cash"

    # fallback: if no explicit TOTAL line found, use the largest money value
    if result["total"] is None:
        amounts = [_clean_money(m) for m in MONEY_RE.findall(full_text)]
        amounts = [a for a in amounts if a is not None]
        if amounts:
            result["total"] = max(amounts)

    return result
