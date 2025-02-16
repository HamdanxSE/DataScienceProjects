import concurrent
import os
import json
import fitz  # PyMuPDF for PDF processing
import re
import time
import logging
from google import genai
from dotenv import load_dotenv
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# ======================
# CONFIGURATION SECTION
# ======================

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=GEMINI_API_KEY)

PDF_BASE_FOLDER = r"H:\DataScrapped"
CATEGORIES = ["Deep Learning", "Computer Vision", "Reinforcement Learning", "NLP", "Optimization"]
OUTPUT_JSON = "annotations.json"

MAX_RETRIES = 3
API_DELAY = 0.5  # ✅ Optimized delay
MAX_WORKERS = 50  # ✅ Run 50 PDFs in parallel

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ======================
# CORE FUNCTIONS
# ======================

def extract_text_from_pdf(pdf_path: str) -> tuple:
    """Extracts the title and abstract from a PDF file using PyMuPDF (fitz)."""
    try:
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
        abstract_patterns = [r'^\s*abstract\b', r'^\s*summary\b']
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

def classify_paper(title):
    """Classify paper using Gemini API and return category + reason."""
    if not title:
        return "Uncategorized", "No title extracted."

    prompt = f"""Classify this research paper into ONE category from this list: {", ".join(CATEGORIES)}.
    Provide the category name followed by a brief reason in the format:

    Category: [Category Name]  
    Reason: [Brief Explanation]

    Title: {title}"""

    for attempt in range(MAX_RETRIES):
        try:
            response = client.models.generate_content(
                model="gemini-1.5-flash",  # ✅ Faster model
                contents=prompt
            )
            response_text = response.text.strip()

            match = re.search(r"Category:\s*(.+)\s*Reason:\s*(.+)", response_text, re.DOTALL)
            if match:
                category, reason = match.groups()
                category, reason = category.strip(), reason.strip()

                if category in CATEGORIES:
                    return category, reason
                return "Uncategorized", "Invalid category response"
        except Exception as e:
            logger.warning(f"API Error (attempt {attempt+1}/{MAX_RETRIES}): {e}. Retrying...")
            time.sleep(API_DELAY * 2)

    return "Classification Failed", "API error or max retries reached"

def save_to_json(data):
    """Append results to JSON file in a structured list format."""
    try:
        if not os.path.exists(OUTPUT_JSON):
            with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
                json.dump([], f)  # Initialize an empty JSON array

        # Load existing data
        with open(OUTPUT_JSON, "r", encoding="utf-8") as f:
            try:
                existing_data = json.load(f)
                if not isinstance(existing_data, list):
                    existing_data = []
            except json.JSONDecodeError:
                existing_data = []

        # Append new data
        existing_data.append(data)

        # Write updated data
        with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
            json.dump(existing_data, f, indent=4)

    except Exception as e:
        logger.error(f"Unexpected JSON error: {e}")

# ======================
# MAIN PROCESSING
# ======================

def process_pdf(pdf_path):

    logger.info(f"Starting: {pdf_path}")

    """Process a single PDF: Extract title, classify, and save result."""
    try:
        title, abstract = extract_text_from_pdf(pdf_path)
        category, reason = classify_paper(title)

        result = {
            "file": os.path.basename(pdf_path),
            "title": title or "Unknown",
            "category": category,
            "reason": reason
        }

        save_to_json(result)
    except Exception as e:
        logger.error(f"Error processing file {pdf_path}: {e}")

    logger.info(f"Completed: {pdf_path}")

def main():
    """Process all PDFs in the folder."""
    pdf_files = [os.path.join(root, f) for root, _, files in os.walk(PDF_BASE_FOLDER)
                 for f in files if f.lower().endswith('.pdf')]

    if not pdf_files:
        logger.error("No PDFs found in the directory.")
        return

    total_files = len(pdf_files)  # ✅ Process all PDFs
    logger.info(f"Processing {total_files} PDFs.")

    results = []
    futures = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_pdf, pdf): pdf for pdf in pdf_files}

        for future in tqdm(as_completed(futures), total=total_files, desc="Processing PDFs"):
            pdf = futures[future]
            try:
                result = future.result()  # Blocks until task completes
                if result:
                    results.append(result)
            except Exception as e:
                logger.error(f"Error processing {pdf}: {e}")

    logger.info(f"\nExpected: {total_files}, Processed: {len(results)}")
    logger.info(f"\nProcessing complete! Results saved to {OUTPUT_JSON}")

if __name__ == "__main__":
    main()




