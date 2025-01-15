import fitz
import pdfplumber
from pdf2image import convert_from_path
import pytesseract
import re
from typing import Optional, List, Dict
import dateutil.parser
import logging
from datetime import datetime
import os
import sys
import platform

class PDFDateExtractor:
    def __init__(self, poppler_path: Optional[str] = None):
        self.date_patterns = [
            r'(?i)(?:expiry date|date of expiry|expires on|valid until|valid through|expiration date)[\s:]+([^\n]*)',
            r'(?i)expiry[\s:]+([^\n]*)',
            r'(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})',
            r'(\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4})',
        ]
        
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)
        
        # Configure Poppler path
        self.poppler_path = self._configure_poppler_path(poppler_path)
        
        # Configure Tesseract path for Windows
        if platform.system() == 'Windows':
            pytesseract.pytesseract.tesseract_cmd = self._get_tesseract_path()

    def _parse_date(self, date_str: str) -> Optional[datetime]:
        """Attempt to parse a date string into a datetime object."""
        try:
            return dateutil.parser.parse(date_str, fuzzy=True)
        except (ValueError, TypeError):
            return None

    def _clean_text(self, text: str) -> str:
        """Clean and normalize extracted text."""
        return ' '.join(text.split())

    def _extract_dates_from_text(self, text: str) -> List[datetime]:
        """Extract and parse dates from text using multiple patterns."""
        dates = []
        for pattern in self.date_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                if len(match.groups()) > 0:
                    date_str = match.group(1)
                    parsed_date = self._parse_date(date_str)
                    if parsed_date:
                        dates.append(parsed_date)
        return dates

    def _get_tesseract_path(self) -> str:
        """Get Tesseract executable path on Windows."""
        default_paths = [
            r'C:\Program Files\Tesseract-OCR\tesseract.exe',
        ]
        
        # First check if it's in PATH
        if os.getenv('TESSERACT_PATH'):
            return os.getenv('TESSERACT_PATH')
            
        # Check default installation paths
        for path in default_paths:
            if os.path.exists(path):
                return path
                
        return 'tesseract'  # Default to expecting it in PATH

    def _configure_poppler_path(self, provided_path: Optional[str] = None) -> Optional[str]:
        """Configure Poppler path based on OS and environment."""
        if provided_path and os.path.exists(provided_path):
            return provided_path

        if platform.system() == 'Windows':
            # Check environment variable
            poppler_env = os.getenv('POPPLER_PATH')
            if poppler_env and os.path.exists(poppler_env):
                return poppler_env

            # Check common Windows installation paths
            common_paths = [
                r'C:\Program Files\poppler-24.08.0\Library\bin',
            ]
            
            for path in common_paths:
                if os.path.exists(path):
                    return path

        return None

    def extract_with_pymupdf(self, pdf_path: str) -> List[datetime]:
        """Extract dates using PyMuPDF."""
        try:
            doc = fitz.open(pdf_path)
            dates = []
            for page in doc:
                text = page.get_text()
                dates.extend(self._extract_dates_from_text(self._clean_text(text)))
            doc.close()
            return dates
        except Exception as e:
            self.logger.error(f"PyMuPDF extraction failed: {str(e)}")
            return []

    def extract_with_pdfplumber(self, pdf_path: str) -> List[datetime]:
        """Extract dates using pdfplumber."""
        try:
            with pdfplumber.open(pdf_path) as pdf:
                dates = []
                for page in pdf.pages:
                    text = page.extract_text()
                    if text:
                        dates.extend(self._extract_dates_from_text(self._clean_text(text)))
                return dates
        except Exception as e:
            self.logger.error(f"pdfplumber extraction failed: {str(e)}")
            return []

    def _check_dependencies(self) -> bool:
        """Check if all required dependencies are available."""
        try:
            # Check Poppler
            if platform.system() == 'Windows' and not self.poppler_path:
                self.logger.error("Poppler not found. Please install Poppler and set its path.")
                return False

            # Check Tesseract
            pytesseract.get_tesseract_version()
            
            return True
        except Exception as e:
            self.logger.error(f"Dependency check failed: {str(e)}")
            return False

    def extract_with_ocr(self, pdf_path: str) -> List[datetime]:
        """Extract dates using OCR."""
        if not self._check_dependencies():
            self.logger.error("Required dependencies not found. Skipping OCR extraction.")
            return []

        try:
            # Convert PDF to images
            convert_kwargs = {}
            if platform.system() == 'Windows':
                convert_kwargs['poppler_path'] = self.poppler_path

            images = convert_from_path(pdf_path, **convert_kwargs)
            
            dates = []
            for image in images:
                text = pytesseract.image_to_string(image)
                dates.extend(self._extract_dates_from_text(self._clean_text(text)))
            return dates
        except Exception as e:
            self.logger.error(f"OCR extraction failed: {str(e)}")
            return []

    def extract_expiry_date(self, pdf_path: str) -> Optional[datetime]:
        """
        Main method to extract expiry date using all available methods.
        Returns the most likely expiry date or None if no date is found.
        """
        if not os.path.exists(pdf_path):
            self.logger.error(f"PDF file not found: {pdf_path}")
            return None

        all_dates = []
        
        # Try all extraction methods
        self.logger.info("Attempting extraction with PyMuPDF...")
        all_dates.extend(self.extract_with_pymupdf(pdf_path))
        
        self.logger.info("Attempting extraction with pdfplumber...")
        all_dates.extend(self.extract_with_pdfplumber(pdf_path))
        
        self.logger.info("Attempting extraction with OCR...")
        all_dates.extend(self.extract_with_ocr(pdf_path))
        
        if not all_dates:
            self.logger.warning("No dates found in the document")
            return None
            
        # Sort dates and return the latest one (assuming it's the expiry date)
        return max(all_dates) if all_dates else None

def main():
    # Example usage with explicit Poppler path (for Windows)
    poppler_path = r"C:\Program Files\poppler-24.08.0\Library\bin"  # Modify this path according to your installation
    
    extractor = PDFDateExtractor(poppler_path=poppler_path)
    pdf_path = "data/doc3.pdf"
    
    expiry_date = extractor.extract_expiry_date(pdf_path)
    if expiry_date:
        print(f"Found expiry date: {expiry_date.strftime('%m-%d-%Y')}")
    else:
        print("No expiry date found in the document")

if __name__ == "__main__":
    main()