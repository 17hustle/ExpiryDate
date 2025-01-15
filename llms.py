import os
from typing import Optional
from datetime import datetime
import logging
from groq import Groq
import fitz  # PyMuPDF
import json
from pdf2image import convert_from_path
import pytesseract
import platform

class HybridDateExtractor:
    def __init__(self, api_key: str, poppler_path: Optional[str] = None):
        """
        Initialize the Hybrid Date Extractor.
        
        Args:
            api_key (str): Groq API key
            poppler_path (str, optional): Path to Poppler binaries for Windows
        """
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)
        
        # Initialize Groq client
        self.client = Groq(api_key=api_key)
        
        # Configure paths
        self.poppler_path = poppler_path
        if platform.system() == 'Windows':
            pytesseract.pytesseract.tesseract_cmd = self._get_tesseract_path()
        
        # System prompt for date extraction
        self.system_prompt = """You are a specialized date extraction assistant. Your task is to:
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

    def _get_tesseract_path(self) -> str:
        """Get Tesseract executable path on Windows."""
        default_paths = [
            r'C:\Program Files\Tesseract-OCR\tesseract.exe',
        ]
        
        if os.getenv('TESSERACT_PATH'):
            return os.getenv('TESSERACT_PATH')
            
        for path in default_paths:
            if os.path.exists(path):
                return path
                
        return 'tesseract'

    def _extract_text_with_ocr(self, pdf_path: str) -> str:
        """
        Extract text from PDF using OCR.
        
        Args:
            pdf_path (str): Path to the PDF file
            
        Returns:
            str: Extracted text content
        """
        try:
            convert_kwargs = {}
            if platform.system() == 'Windows' and self.poppler_path:
                convert_kwargs['poppler_path'] = self.poppler_path

            images = convert_from_path(pdf_path, **convert_kwargs)
            
            text_content = []
            for image in images:
                text = pytesseract.image_to_string(image)
                text_content.append(text)
            
            return "\n".join(text_content)
        except Exception as e:
            self.logger.error(f"OCR extraction failed: {str(e)}")
            return ""

    def _extract_text_from_pdf(self, pdf_path: str) -> str:
        """
        Extract text from PDF using both PyMuPDF and OCR if needed.
        
        Args:
            pdf_path (str): Path to the PDF file
            
        Returns:
            str: Extracted text content
        """
        try:
            # Try PyMuPDF first
            doc = fitz.open(pdf_path)
            text = ""
            for page in doc:
                text += page.get_text()
            doc.close()

            # If no text was extracted, try OCR
            if not text.strip():
                self.logger.info("No text extracted with PyMuPDF, trying OCR...")
                text = self._extract_text_with_ocr(pdf_path)

            return text
        except Exception as e:
            self.logger.error(f"Error extracting text from PDF: {str(e)}")
            return ""

    def _parse_llm_response(self, response: str) -> Optional[datetime]:
        """Parse the LLM response and convert it to a datetime object."""
        try:
            data = json.loads(response)
            date_str = data.get('expiry_date')
            if date_str:
                return datetime.fromisoformat(date_str)
            return None
        except (json.JSONDecodeError, ValueError) as e:
            self.logger.error(f"Error parsing LLM response: {str(e)}")
            return None

    def extract_date(self, pdf_path: str) -> Optional[datetime]:
        """Extract expiry date from a PDF using hybrid approach."""
        try:
            # Extract text using both methods
            text_content = self._extract_text_from_pdf(pdf_path)
            if not text_content:
                self.logger.error("No text content extracted from PDF")
                return None

            self.logger.info("Extracted text content, sending to LLM for analysis...")
            
            # Debug: Print extracted text
            self.logger.debug(f"Extracted text: {text_content}")
            
            # Prepare the user prompt with the extracted text
            user_prompt = f"""Extract the expiry date from this text. Remember to return ONLY a JSON response:

{text_content}"""

            # Get LLM response using synchronous create method
            chat_completion = self.client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                max_tokens=100
            )

            # Parse and return the response
            response_text = chat_completion.choices[0].message.content
            self.logger.debug(f"LLM response: {response_text}")
            return self._parse_llm_response(response_text)

        except Exception as e:
            self.logger.error(f"Error in date extraction: {str(e)}")
            return None

def main():
    # Initialize extractor with your API key
    api_key = "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"
    if not api_key:
        raise ValueError("GROQ_API_KEY environment variable not set")
    
    # For Windows, specify the Poppler path
    poppler_path = r"C:\Program Files\poppler-24.08.0\Library\bin"
    
    extractor = HybridDateExtractor(api_key, poppler_path)
    
    # Example usage
    pdf_path = "data/doc2.pdf"  # Update with your PDF path
    date = extractor.extract_date(pdf_path)
    
    if date:
        print(f"Found expiry date: {date.strftime('%Y-%m-%d')}")
    else:
        print("No expiry date found in the document")

if __name__ == "__main__":
    main()