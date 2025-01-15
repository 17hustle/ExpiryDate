import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import os
from typing import Optional, Dict, List
import platform
from pathlib import Path
import tempfile
import sys
import logging
from groq import Groq
import fitz  # PyMuPDF
import json
from pdf2image import convert_from_path
import pytesseract

# First, let's include the HybridDateExtractor class
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
        """Extract text from PDF using OCR."""
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
        """Extract text from PDF using both PyMuPDF and OCR if needed."""
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
            
            # Prepare the user prompt with the extracted text
            user_prompt = f"""Extract the expiry date from this text. Remember to return ONLY a JSON response:

{text_content}"""

            # Get LLM response
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
            return self._parse_llm_response(response_text)

        except Exception as e:
            self.logger.error(f"Error in date extraction: {str(e)}")
            return None

# Configure page settings
st.set_page_config(
    page_title="Document Expiry Tracker",
    page_icon="📄",
    layout="wide"
)
# Initialize session state
if 'documents_data' not in st.session_state:
    st.session_state.documents_data = []

def check_dependencies() -> tuple[bool, str]:
    """Check if required dependencies are installed and configured."""
    issues = []
    
    # Check Tesseract
    try:
        import pytesseract
        if platform.system() == 'Windows':
            if not os.path.exists(r'C:\Program Files\Tesseract-OCR\tesseract.exe'):
                issues.append("Tesseract OCR not found in default Windows location")
    except ImportError:
        issues.append("pytesseract not installed")
    
    # Check Poppler
    if platform.system() == 'Windows':
        poppler_found = False
        common_paths = [
            r"C:\Program Files\poppler-24.08.0\Library\bin",
            r"C:\Program Files\poppler\bin",
            r"C:\poppler\bin"
        ]
        for path in common_paths:
            if os.path.exists(path):
                poppler_found = True
                break
        if not poppler_found:
            issues.append("Poppler not found in common Windows locations")
    
    return (len(issues) == 0, "\n".join(issues))

def calculate_status(expiry_date: datetime) -> tuple[str, int]:
    """Calculate document status and days until expiry."""
    today = datetime.now().date()
    expiry_date = expiry_date.date() if isinstance(expiry_date, datetime) else expiry_date
    days_until_expiry = (expiry_date - today).days
    
    if days_until_expiry < 0:
        return "Overdue", days_until_expiry
    elif days_until_expiry <= 30:
        return "Expiring Soon", days_until_expiry
    else:
        return "Up-to-date", days_until_expiry

def document_exists(filename: str) -> bool:
    """Check if a document already exists in session state."""
    return any(doc['filename'] == filename for doc in st.session_state.documents_data)

def process_documents(uploaded_files: list, extractor: HybridDateExtractor) -> List[Dict]:
    """Process uploaded documents and extract expiry dates."""
    results = []
    
    for uploaded_file in uploaded_files:
        # Skip if document already exists
        if document_exists(uploaded_file.name):
            continue
            
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
        
        try:
            # Extract date using the hybrid extractor
            extracted_date = extractor.extract_date(tmp_path)
            
            # Calculate status and days until expiry
            status, days = ("Date Required", 0) if not extracted_date else calculate_status(extracted_date)
            
            results.append({
                "filename": uploaded_file.name,
                "expiry_date": extracted_date,
                "status": status,
                "days_until_expiry": days
            })
        finally:
            # Clean up temporary file
            os.unlink(tmp_path)
    
    return results

def display_documents_table(df: pd.DataFrame, tab_name: str):
    """Display and handle the interactive documents table."""
    if len(df) == 0:
        st.info("No documents in this category.")
        return
    
    # Create a copy of the DataFrame for display
    display_df = df.copy()
    
    # Format dates and create edit columns
    display_df['expiry_date'] = display_df['expiry_date'].apply(
        lambda x: x.strftime('%Y-%m-%d') if pd.notnull(x) else 'Not Set'
    )
    
    # Convert days to more readable format
    display_df['days_until_expiry'] = display_df['days_until_expiry'].apply(
        lambda x: f"{abs(x)} days {'overdue' if x < 0 else 'remaining'}" if x != 0 else "Date Required"
    )
    
    # Apply color coding to status
    display_df['status'] = display_df['status'].apply(
        lambda x: color_code_status(x)
    )
    
    # Display the table
    st.write(display_df)
    
    # Add edit functionality
    st.subheader("Update Expiry Date")
    col1, col2 = st.columns(2)
    
    with col1:
        selected_doc = st.selectbox(
            "Select document to update",
            options=df['filename'].tolist(),
            key=f"select_{tab_name}"
        )
    
    with col2:
        new_date = st.date_input(
            "New expiry date",
            min_value=datetime.now().date(),
            key=f"date_{tab_name}"
        )
    
    if st.button("Update Date", key=f"button_{tab_name}"):
        # Find the document in session state and update it
        for doc in st.session_state.documents_data:
            if doc['filename'] == selected_doc:
                doc['expiry_date'] = datetime.combine(new_date, datetime.min.time())
                status, days = calculate_status(doc['expiry_date'])
                doc['status'] = status
                doc['days_until_expiry'] = days
                break
        st.success(f"Updated expiry date for {selected_doc}")
        st.rerun()

def color_code_status(status: str) -> str:
    """Apply color coding to status text."""
    if status == "Up-to-date":
        return "🟢 " + status
    elif status == "Expiring Soon":
        return "🟡 " + status
    elif status == "Overdue":
        return "🔴 " + status
    return "⚪ " + status

def main():
    st.title("📄 Document Expiry Tracker")
    
    # Check dependencies
    deps_ok, issues = check_dependencies()
    if not deps_ok:
        st.error("⚠️ System Configuration Issues Detected:")
        st.code(issues)
        st.stop()
    
    # Initialize the extractor
    api_key = st.secrets.get("GROQ_API_KEY")
    if not api_key:
        st.error("❌ GROQ API key not configured in Streamlit secrets!")
        st.stop()
    
    poppler_path = None
    if platform.system() == 'Windows':
        poppler_path = r"C:\Program Files\poppler-24.08.0\Library\bin"
    
    extractor = HybridDateExtractor(api_key, poppler_path)
    
    # File uploader
    uploaded_files = st.file_uploader(
        "Upload PDF documents",
        type=['pdf'],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        with st.spinner('Processing documents...'):
            new_documents = process_documents(uploaded_files, extractor)
            if new_documents:  # Only extend if there are actually new documents
                st.session_state.documents_data.extend(new_documents)
                st.success(f"Added {len(new_documents)} new document(s)")
            else:
                st.info("No new documents to add")
    
    # Display documents table
    if st.session_state.documents_data:
        st.subheader("Document Status Dashboard")
        
        # Convert to DataFrame for easier manipulation
        df = pd.DataFrame(st.session_state.documents_data)
        
        # Create tabs for different views
        tab1, tab2, tab3, tab4 = st.tabs([
            "All Documents",
            "Up-to-date",
            "Expiring Soon",
            "Overdue"
        ])
        
        with tab1:
            display_documents_table(df, "all")
        
        with tab2:
            display_documents_table(df[df['status'] == 'Up-to-date'], "uptodate")
            
        with tab3:
            display_documents_table(df[df['status'] == 'Expiring Soon'], "expiring")
            
        with tab4:
            display_documents_table(df[df['status'] == 'Overdue'], "overdue")
        
        # Add clear button
        if st.button("Clear All Documents"):
            st.session_state.documents_data = []
            st.experimental_rerun()

if __name__ == "__main__":
    main()