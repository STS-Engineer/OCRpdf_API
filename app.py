import logging
import time
from pathlib import Path
import contextlib

from flask import Flask, request, jsonify
import nltk
import torch

from pdf2text import *

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

app = Flask(__name__)

nltk.download("stopwords")

# Initialize OCR model globally
ocr_model = None


def init_ocr_model():
    """Initialize the OCR model"""
    global ocr_model
    logging.info("Loading OCR model")
    with contextlib.redirect_stdout(None):
        ocr_model = ocr_predictor(
            "db_resnet50",
            "crnn_mobilenet_v3_large",
            pretrained=True,
            assume_straight_pages=True,
        )


def convert_PDF(pdf_path, language: str = "en", max_pages=20):
    """
    convert_PDF - convert a PDF file to text

    Args:
        pdf_path (str): Path to PDF file
        language (str, optional): Language to use for OCR. Defaults to "en".
        max_pages (int): Maximum number of pages to process

    Returns:
        dict: Conversion results
    """
    rm_local_text_files()
    global ocr_model
    
    st = time.perf_counter()
    file_path = Path(pdf_path)
    
    # Vérifier si le fichier existe
    if not file_path.exists():
        logging.error(f"File {file_path} does not exist")
        return {
            "success": False,
            "error": f"File not found: {pdf_path}",
            "text": None
        }
    
    if not file_path.suffix.lower() == ".pdf":
        logging.error(f"File {file_path} is not a PDF file")
        return {
            "success": False,
            "error": "File is not a PDF file. Please provide a PDF file path.",
            "text": None
        }

    try:
        conversion_stats = convert_PDF_to_Text(
            file_path,
            ocr_model=ocr_model,
            max_pages=max_pages,
        )
        
        converted_txt = conversion_stats["converted_text"]
        num_pages = conversion_stats["num_pages"]
        was_truncated = conversion_stats["truncated"]

        rt = round((time.perf_counter() - st) / 60, 2)
        logging.info(f"Runtime: {rt} minutes")

        return {
            "success": True,
            "text": converted_txt,
            "runtime_minutes": rt,
            "num_pages": num_pages,
            "was_truncated": was_truncated,
            "max_pages": max_pages,
            "pdf_path": str(file_path)
        }
        
    except Exception as e:
        logging.error(f"Error converting PDF: {e}")
        return {
            "success": False,
            "error": str(e),
            "text": None
        }


@app.route('/')
def index():
    """API info endpoint"""
    return jsonify({
        "message": "PDF OCR API",
        "version": "2.0",
        "endpoints": {
            "/convert": {
                "method": "POST",
                "description": "Convert PDF to text by providing file path",
                "parameters": {
                    "pdf_path": "Full path to PDF file (required, JSON body)",
                    "max_pages": "Maximum pages to process (optional, default: 20)"
                },
                "example": {
                    "pdf_path": "C:\\Users\\username\\Desktop\\document.pdf",
                    "max_pages": 20
                }
            },
            "/health": {
                "method": "GET",
                "description": "Check API health"
            }
        }
    })


@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "ocr_model_loaded": ocr_model is not None,
        "gpu_available": torch.cuda.is_available()
    })


@app.route('/convert', methods=['POST'])
def convert():
    """
    Handle PDF conversion by file path
    
    POST /convert
    Content-Type: application/json
    Body:
        {
            "pdf_path": "C:\\path\\to\\file.pdf",
            "max_pages": 20  (optional)
        }
    """
    # Vérifier que le body est du JSON
    if not request.is_json:
        return jsonify({
            "success": False, 
            "error": "Content-Type must be application/json"
        }), 400
    
    data = request.get_json()
    
    # Vérifier que pdf_path est fourni
    if 'pdf_path' not in data:
        return jsonify({
            "success": False, 
            "error": "Missing 'pdf_path' parameter in JSON body"
        }), 400
    
    pdf_path = data['pdf_path']
    
    if not pdf_path or pdf_path.strip() == '':
        return jsonify({
            "success": False, 
            "error": "pdf_path cannot be empty"
        }), 400
    
    try:
        # Get max_pages parameter (default: 20)
        max_pages = int(data.get('max_pages', 20))
        
        logging.info(f"Converting PDF: {pdf_path}")
        
        # Convert PDF
        result = convert_PDF(pdf_path, max_pages=max_pages)
        
        return jsonify(result)
        
    except Exception as e:
        logging.error(f"Error in convert endpoint: {e}")
        return jsonify({
            "success": False, 
            "error": f"Server error: {str(e)}"
        }), 500


if __name__ == "__main__":
    logging.info("Starting Flask PDF OCR API")
    
    use_GPU = torch.cuda.is_available()
    logging.info(f"Using GPU: {use_GPU}")
    
    # Initialize OCR model
    init_ocr_model()
    
    logging.info("Flask API ready")
    logging.info("Access API at: http://localhost:5000")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
