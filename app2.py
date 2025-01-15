import os
import json
import logging
from datetime import datetime
from typing import Optional
import fitz  # PyMuPDF
from pdf2image import convert_from_path
import pytesseract
import platform
import streamlit as st
from groq import Groq

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Hardcoded Groq API Key (replace with your actual key)
GROQ_API_KEY = "gsk_your_api_key_here"

# Configure Tesseract path
if platform.system() == 'Windows':
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
else:
    pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'  # Default path for Linux/macOS

# Configure Poppler path
if platform.system() == 'Windows':
    poppler_path = r'C:\poppler-24.02.0\Library\bin'  # Update this path to your Poppler installation
else:
    poppler_path = None  # Use default path for Linux/macOS

# Initialize Groq client
def initialize_groq_client():
    return Groq(api_key=GROQ_API_KEY)

# System prompt for date extraction
SYSTEM_PROMPT = """You are a specialized date extraction assistant. Your task is to:
1. Extract ONLY the expiry date or date of expiry from the given text
2. Ignore all other information in the document
3. Return the date in ISO format (YYYY-MM-DD)
4. If multiple dates are found, identify which one is the expiry date based on context
5. Look for key phrases like "expiry date", "valid until", "expires on", "expiration date"
6. Return null if no expiry date is found
7. Return your response in valid JSON format with a single key 'expiry_date'

Example input/output pairs:
Input: "Issue Date: 2023-01-01 Expiry Date: 2025-12-31"
Output: {"expiry_date": "2025-12-31"}

Input: "Valid from 2024-01-01 until 2026-01-01"
Output: {"expiry_date": "2026-01-01"}

Input: "Document issued on 2024-03-15"
Output: {"expiry_date": null}

Return ONLY the JSON response, no additional text."""

# Extract text from PDF using PyMuPDF and OCR
def extract_text_from_pdf(pdf_path: str, poppler_path: Optional[str] = None) -> str:
    try:
        # Try PyMuPDF first
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()

        # If no text was extracted, try OCR
        if not text.strip():
            logger.info("No text extracted with PyMuPDF, trying OCR...")
            convert_kwargs = {}
            if platform.system() == 'Windows' and poppler_path:
                convert_kwargs['poppler_path'] = poppler_path

            images = convert_from_path(pdf_path, **convert_kwargs)
            text_content = []
            for image in images:
                text = pytesseract.image_to_string(image)
                text_content.append(text)
            text = "\n".join(text_content)

        return text
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {str(e)}")
        return ""

# Parse LLM response and convert to datetime
def parse_llm_response(response: str) -> Optional[datetime]:
    try:
        data = json.loads(response)
        date_str = data.get('expiry_date')
        if date_str:
            return datetime.fromisoformat(date_str)
        return None
    except (json.JSONDecodeError, ValueError) as e:
        logger.error(f"Error parsing LLM response: {str(e)}")
        return None

# Extract expiry date using Groq API
def extract_expiry_date(client, text_content: str) -> Optional[datetime]:
    try:
        user_prompt = f"""Extract the expiry date from this text. Remember to return ONLY a JSON response:

{text_content}"""

        chat_completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1,
            max_tokens=100
        )

        response_text = chat_completion.choices[0].message.content
        logger.debug(f"LLM response: {response_text}")
        return parse_llm_response(response_text)
    except Exception as e:
        logger.error(f"Error in date extraction: {str(e)}")
        return None

# Streamlit App
def main():
    st.title("📄 PDF Expiry Date Extractor")
    st.write("Upload a PDF to extract the expiry date. If the model cannot extract a date, you can manually input it.")

    # Initialize Groq client
    client = initialize_groq_client()

    # Upload PDF
    uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")
    if not uploaded_file:
        st.info("Please upload a PDF file.")
        return

    # Save the uploaded file temporarily
    temp_file_path = f"temp_{uploaded_file.name}"
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Extract text from PDF
    text_content = extract_text_from_pdf(temp_file_path, poppler_path)
    if not text_content:
        st.error("Failed to extract text from the PDF.")
        os.remove(temp_file_path)
        return

    # Extract expiry date using Groq API
    expiry_date = extract_expiry_date(client, text_content)
    if expiry_date:
        st.success(f"Extracted expiry date: {expiry_date.strftime('%Y-%m-%d')}")
    else:
        st.warning("No expiry date found. Please enter it manually.")

        # Manual input for missing dates
        manual_date = st.date_input("Enter expiry date manually:")
        if manual_date:
            expiry_date = manual_date
            st.success(f"Manual expiry date saved: {expiry_date.strftime('%Y-%m-%d')}")

    # Clean up temporary file
    os.remove(temp_file_path)

    # Display the final result
    if expiry_date:
        st.subheader("Final Result")
        st.write(f"**Expiry Date:** {expiry_date.strftime('%Y-%m-%d')}")

# Run the Streamlit app
if __name__ == "__main__":
    main()