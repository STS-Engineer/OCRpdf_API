import logging
import time
from pathlib import Path
import contextlib
import os
from werkzeug.utils import secure_filename

from flask import Flask, request, jsonify
import nltk
import torch

from pdf2text import *

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

app = Flask(__name__)

# Configure upload settings - Azure-friendly path
UPLOAD_FOLDER = Path('/home/site/wwwroot/uploads')
UPLOAD_FOLDER.mkdir(exist_ok=True, parents=True)
ALLOWED_EXTENSIONS = {'pdf', 'png', 'jpg', 'jpeg', 'tiff', 'bmp'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size

nltk.download("stopwords")

# Initialize OCR model globally
ocr_model = None


def init_ocr_model():
    """Initialize the OCR model"""
    global ocr_model
    logging.info("Loading OCR model...")
    with contextlib.redirect_stdout(None):
        ocr_model = ocr_predictor(
            "db_resnet50",
            "crnn_mobilenet_v3_large",
            pretrained=True,
            assume_straight_pages=True,
        )
    logging.info("OCR model loaded successfully into the application process.")


try:
    init_ocr_model()
except Exception as e:
    logging.error(f"FATAL: Failed to load OCR model during Gunicorn preload: {e}", exc_info=True)


def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def convert_PDF(pdf_path, language: str = "en", max_pages=20):
    """
    Convert PDF/image file to text using OCR
    """
    rm_local_text_files()
    global ocr_model
    
    st = time.perf_counter()
    file_path = Path(pdf_path)
    
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
        "version": "3.0",
        "endpoints": {
            "/upload": {
                "method": "POST",
                "description": "Upload and convert PDF/image to text",
                "parameters": {
                    "file": "PDF or image file (required, form-data)",
                    "max_pages": "Maximum pages to process (optional, default: 20)"
                },
                "example": "Use multipart/form-data with 'file' field"
            },
            "/convert": {
                "method": "POST",
                "description": "Convert PDF to text by providing file path (server-side files only)",
                "parameters": {
                    "pdf_path": "Full path to PDF file (required, JSON body)",
                    "max_pages": "Maximum pages to process (optional, default: 20)"
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
        "gpu_available": torch.cuda.is_available(),
        "upload_folder": str(UPLOAD_FOLDER),
        "upload_folder_exists": UPLOAD_FOLDER.exists()
    })


@app.route('/upload', methods=['POST'])
def upload_and_convert():
    """
    Upload PDF/image file and convert to text
    
    POST /upload
    Content-Type: multipart/form-data
    Form data:
        - file: PDF or image file (required)
        - max_pages: Maximum pages to process (optional, default: 20)
    """
    # Check if file is in request
    if 'file' not in request.files:
        return jsonify({
            "success": False,
            "error": "No file provided. Please upload a file using 'file' field in form-data"
        }), 400
    
    file = request.files['file']
    
    # Check if file is selected
    if file.filename == '':
        return jsonify({
            "success": False,
            "error": "No file selected"
        }), 400
    
    # Check file extension
    if not allowed_file(file.filename):
        return jsonify({
            "success": False,
            "error": f"Invalid file type. Allowed types: {', '.join(ALLOWED_EXTENSIONS)}"
        }), 400
    
    try:
        # Secure the filename and save
        filename = secure_filename(file.filename)
        filepath = UPLOAD_FOLDER / filename
        
        # Save uploaded file
        file.save(str(filepath))
        logging.info(f"File saved to: {filepath}")
        
        # Get max_pages parameter
        max_pages = int(request.form.get('max_pages', 20))
        
        # Convert the file
        result = convert_PDF(str(filepath), max_pages=max_pages)
        
        # Add upload info to result
        result['uploaded_filename'] = filename
        result['saved_path'] = str(filepath)
        
        # Optional: Clean up file after processing
        # os.remove(filepath)
        
        return jsonify(result)
        
    except Exception as e:
        logging.error(f"Error in upload endpoint: {e}")
        return jsonify({
            "success": False,
            "error": f"Server error: {str(e)}"
        }), 500


@app.route('/convert', methods=['POST'])
def convert():
    """
    Handle PDF conversion by file path (for files already on server)
    
    POST /convert
    Content-Type: application/json
    Body:
        {
            "pdf_path": "/path/to/file.pdf",
            "max_pages": 20 (optional)
        }
    """
    if not request.is_json:
        return jsonify({
            "success": False, 
            "error": "Content-Type must be application/json"
        }), 400
    
    data = request.get_json()
    
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
    logging.info(f"Upload folder: {UPLOAD_FOLDER}")
    
    logging.info("Flask API ready")
    logging.info("Access API at: http://localhost:5000")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
