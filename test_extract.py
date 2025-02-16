

import os
import fitz  # PyMuPDF
import re
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_text_from_pdf(pdf_path: str) -> tuple:
    """Extracts the title and abstract from a PDF file."""
    try:
        # Open PDF and extract text from first two pages
        doc = fitz.open(pdf_path)
        text = []
        for page in doc[:2]:  # Process first two pages
            text.append(page.get_text("text"))
        full_text = "\n".join(text)

        # Split into lines and clean empty spaces
        lines = [line.strip() for line in full_text.split("\n") if line.strip()]

        if not lines:
            return None, None

        # Extract title: First valid text line that looks like a title
        title = None
        for line in lines[:5]:  # Check first few lines for title
            if len(line) > 5 and not re.match(r'^\d+\.?$', line):  # Skip numbers or very short lines
                title = line
                break

        # Identify abstract start position
        abstract_start = None
        abstract_patterns = [
            r'^\s*abstract\b',
            r'^\s*summary\b'
        ]
        for i, line in enumerate(lines[:50]):  # Limit search to first 50 lines
            if any(re.match(p, line, re.IGNORECASE) for p in abstract_patterns):
                abstract_start = i + 1  # Move to next line for content
                break

        # Extract abstract text
        abstract = []
        if abstract_start:
            section_break_pattern = re.compile(r'^\s*(\d+\.?\s*|I\.|INTRODUCTION|RELATED WORK|METHODS?)\s*$', re.IGNORECASE)
            for line in lines[abstract_start:]:
                if section_break_pattern.match(line):  # Stop at the next major section
                    break
                abstract.append(line)

        # Fallback if abstract extraction failed
        if not abstract and len(lines) > 1:
            abstract = lines[1:6]  # Take first 5 lines after title as a rough fallback

        return title, ' '.join(abstract[:500]).strip()

    except Exception as e:
        logger.error(f"Error processing {pdf_path}: {str(e)}")
        return None, None


# Configuration
pdf_folder_2023 = r"H:\DataScrapped\2023"

# Find up to 10 PDFs in the folder
pdf_files = [f for f in os.listdir(pdf_folder_2023) if f.lower().endswith(".pdf")][:10]

if pdf_files:
    print(f"\nProcessing {len(pdf_files)} PDFs...\n")

    for i, pdf_file in enumerate(pdf_files, 1):
        pdf_path = os.path.join(pdf_folder_2023, pdf_file)
        print(f"\n[{i}/{len(pdf_files)}] Extracting from: {pdf_file}")

        title, abstract = extract_text_from_pdf(pdf_path)

        print("\n--- Extraction Results ---")
        print(f"Title: {title or 'NOT FOUND'}")
        print(f"\nAbstract: {abstract or 'NOT FOUND'}\n")
        print("=" * 80)

else:
    print("No PDFs found in the 2023 folder!")
