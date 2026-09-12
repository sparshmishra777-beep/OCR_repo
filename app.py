import streamlit as st
from PIL import Image, ImageOps
import pytesseract
import json

from parser import parse_receipt_text


# ============================================================
# TESSERACT CONFIGURATION
# ============================================================

# Windows:
# Change this path if Tesseract is installed somewhere else.
pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Receipt OCR & Information Extraction",
    page_icon="🧾",
    layout="wide"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

        .main-title {
            font-size: 36px;
            font-weight: 700;
            margin-bottom: 5px;
        }

        .subtitle {
            color: #666;
            font-size: 17px;
            margin-bottom: 25px;
        }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def preprocess(img: Image.Image) -> Image.Image:
    """
    Preprocess the receipt image before OCR.

    Steps:
    1. Convert image to grayscale.
    2. Improve contrast using autocontrast.
    """

    img = img.convert("L")

    img = ImageOps.autocontrast(img)

    return img


# ============================================================
# OCR FUNCTION
# ============================================================

def perform_ocr(img: Image.Image) -> str:
    """
    Run Tesseract OCR on the preprocessed image.
    """

    text = pytesseract.image_to_string(
        img,
        config="--psm 6"
    )

    return text


# ============================================================
# COMPLETE RECEIPT PIPELINE
# ============================================================

def process_receipt(image: Image.Image):
    """
    Complete pipeline:

    Image
       ↓
    Preprocessing
       ↓
    Tesseract OCR
       ↓
    Rule-based Parser
       ↓
    Structured JSON
    """

    # Step 1: preprocessing
    processed_image = preprocess(image)

    # Step 2: OCR
    raw_text = perform_ocr(processed_image)

    # Step 3: parse OCR text
    result = parse_receipt_text(
        raw_text,
        source_file="uploaded_receipt"
    )

    return processed_image, raw_text, result


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">'
    '🧾 Receipt OCR & Information Extraction'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="subtitle">
    Convert receipt images into structured financial information
    using Tesseract OCR and rule-based information extraction.
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ Pipeline")

    st.markdown(
        """
        **Current Pipeline**

        1. Receipt Image
        2. PIL Preprocessing
        3. Tesseract OCR
        4. Regex + Keyword Parsing
        5. Structured JSON
        """
    )

    st.divider()

    st.info(
        "The current interface uses the rule-based baseline "
        "pipeline. Donut remains available separately as the "
        "trainable document understanding approach."
    )


# ============================================================
# FILE UPLOADER
# ============================================================

uploaded_file = st.file_uploader(
    "Upload a receipt image",
    type=[
        "jpg",
        "jpeg",
        "png",
        "webp"
    ]
)


# ============================================================
# PROCESS UPLOADED IMAGE
# ============================================================

if uploaded_file is not None:

    # Open uploaded image
    image = Image.open(uploaded_file)

    st.subheader("📷 Uploaded Receipt")

    # Create two columns
    col1, col2 = st.columns(2)

    # --------------------------------------------------------
    # ORIGINAL IMAGE
    # --------------------------------------------------------

    with col1:

        st.image(
            image,
            caption="Original Receipt",
            use_container_width=True
        )

    # --------------------------------------------------------
    # PREPROCESSED IMAGE
    # --------------------------------------------------------

    with col2:

        processed_image = preprocess(image)

        st.image(
            processed_image,
            caption="Preprocessed Image",
            use_container_width=True
        )


    # ========================================================
    # EXTRACT BUTTON
    # ========================================================

    if st.button(
        "🔍 Extract Receipt Information",
        type="primary",
        use_container_width=True
    ):

        with st.spinner("Processing receipt..."):

            try:

                # Run complete pipeline
                processed_image, raw_text, result = process_receipt(
                    image
                )

                # Save results in Streamlit session
                st.session_state["raw_text"] = raw_text

                st.session_state["result"] = result

                st.success(
                    "Receipt processed successfully!"
                )

            except Exception as e:

                st.error(
                    "An error occurred while processing the receipt:"
                )

                st.exception(e)


# ============================================================
# DISPLAY RESULTS
# ============================================================

if "result" in st.session_state:

    result = st.session_state["result"]

    raw_text = st.session_state["raw_text"]

    st.divider()

    st.header("📊 Extracted Information")


    # ========================================================
    # BASIC INFORMATION
    # ========================================================

    col1, col2, col3 = st.columns(3)


    # Merchant
    with col1:

        st.markdown("**Merchant**")

        st.write(
            result.get("merchant_name")
            or "Not detected"
        )


    # Date
    with col2:

        st.markdown("**Date**")

        st.write(
            result.get("date")
            or "Not detected"
        )


    # Time
    with col3:

        st.markdown("**Time**")

        st.write(
            result.get("time")
            or "Not detected"
        )


    # ========================================================
    # FINANCIAL INFORMATION
    # ========================================================

    st.subheader("💰 Financial Information")

    col1, col2, col3, col4 = st.columns(4)


    # Subtotal
    with col1:

        subtotal = result.get("subtotal")

        st.metric(
            "Subtotal",
            subtotal if subtotal is not None else "N/A"
        )


    # Tax
    with col2:

        tax = result.get("tax")

        st.metric(
            "Tax",
            tax if tax is not None else "N/A"
        )


    # Total
    with col3:

        total = result.get("total")

        st.metric(
            "Total",
            total if total is not None else "N/A"
        )


    # Currency
    with col4:

        currency = result.get("currency")

        st.metric(
            "Currency",
            currency if currency else "N/A"
        )


    # ========================================================
    # PAYMENT INFORMATION
    # ========================================================

    st.subheader("💳 Payment Information")

    col1, col2, col3 = st.columns(3)


    # Payment Method
    with col1:

        st.write("**Payment Method**")

        st.write(
            result.get("payment_method")
            or "Not detected"
        )


    # Cash Tendered
    with col2:

        st.write("**Cash Tendered**")

        cash = result.get("cash_tendered")

        st.write(
            cash if cash is not None else "N/A"
        )


    # Change Due
    with col3:

        st.write("**Change Due**")

        change = result.get("change_due")

        st.write(
            change if change is not None else "N/A"
        )


    # ========================================================
    # LINE ITEMS
    # ========================================================

    st.subheader("🛒 Line Items")

    items = result.get("line_items", [])


    if items:

        st.table(items)

    else:

        st.info(
            "No line items detected."
        )


    # ========================================================
    # RAW OCR TEXT
    # ========================================================

    st.subheader("🔤 Raw OCR Text")

    with st.expander("Show OCR output"):

        st.text(raw_text)


    # ========================================================
    # STRUCTURED JSON
    # ========================================================

    st.subheader("🧩 Structured JSON")

    st.json(result)


    # ========================================================
    # DOWNLOAD JSON
    # ========================================================

    json_data = json.dumps(
        result,
        indent=2,
        ensure_ascii=False
    )

    st.download_button(
        label="⬇️ Download JSON",
        data=json_data,
        file_name="receipt_result.json",
        mime="application/json"
    )