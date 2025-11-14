import logging
import time
from pathlib import Path
import contextlib
import os
from werkzeug.utils import secure_filename # Used for safe filename handling

from flask import Flask, request, jsonify
import nltk
import torch

# Assuming pdf2text.py is in the same directory and contains
# rm_local_text_files, convert_PDF_to_Text, and ocr_predictor
from pdf2text import * # --- CONFIGURATION ---
# Use /tmp for transient storage, which is standard practice on Azure App Services
UPLOAD_FOLDER = Path('/tmp/uploads')
ALLOWED_EXTENSIONS = {'pdf', 'png', 'jpg', 'jpeg', 'tiff', 'bmp'}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

app = Flask(__name__)

# Ensure the upload directory exists when the app starts
# This is safe and idempotent (exist_ok=True)
UPLOAD_FOLDER.mkdir(exist_ok=True, parents=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# Set max file size to 50MB
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024

# Initialize NLTK data
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

# Pre-load model at module level for Gunicorn
try:
    init_ocr_model()
except Exception as e:
    logging.error(f"FATAL: Failed to load OCR model during Gunicorn preload: {e}", exc_info=True)


def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# --- REFACTORED CONVERSION CORE ---
def convert_PDF(filename: str, max_pages=20):
    """
    Core function to convert a file located in UPLOAD_FOLDER to text.
    
    Args:
        filename (str): The name of the file saved in the UPLOAD_FOLDER.
    """
    rm_local_text_files()
    global ocr_model
    
    st = time.perf_counter()
    # Construct the full path using the UPLOAD_FOLDER base
    file_path = UPLOAD_FOLDER / filename
    
    if not file_path.exists():
        logging.error(f"File {file_path} does not exist")
        # Raise an error that can be caught by the endpoint
        raise FileNotFoundError(f"File '{filename}' not found on server. Did you call /upload first?")
    
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
            "filename": filename
        }
        
    except Exception as e:
        logging.error(f"Error converting file: {e}")
        # Raise RuntimeError to be caught by the endpoint for a 500 status
        raise RuntimeError(f"OCR processing failed: {str(e)}")


# --- INFO & HEALTH ENDPOINTS ---
@app.route('/')
def index():
    """API info endpoint"""
    return jsonify({
        "message": "PDF OCR API (Two-Step File Processor)",
        "version": "4.0",
        "endpoints": {
            "/upload": {
                "method": "POST",
                "description": "Step 1: Uploads file to server, returns filename for /convert",
            },
            "/convert": {
                "method": "POST",
                "description": "Step 2: Converts the file saved by /upload, returns text, and deletes the file.",
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

# ----------------------------------------------------------------------
# >>> STEP 1: UPLOAD ROUTE (Saves file, returns filename) <<<
# ----------------------------------------------------------------------
@app.route('/upload', methods=['POST'])
def upload_file():
    """
    Handles file upload, saves the file to UPLOAD_FOLDER, and returns the filename.
    """
    if 'file' not in request.files:
        return jsonify({
            "success": False,
            "error": "No file provided. Please upload a file using 'file' field in form-data"
        }), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({
            "success": False,
            "error": "No file selected"
        }), 400
    
    if not allowed_file(file.filename):
        return jsonify({
            "success": False,
            "error": f"Invalid file type. Allowed types: {', '.join(ALLOWED_EXTENSIONS)}"
        }), 400
    
    try:
        filename = secure_filename(file.filename)
        filepath = UPLOAD_FOLDER / filename
        
        # Save uploaded file
        # Using str(filepath) is necessary for older Python versions/libraries
        file.save(str(filepath)) 
        logging.info(f"File successfully uploaded and saved to: {filepath}")
        
        # Return the filename—the key for the next API call
        return jsonify({
            "success": True,
            "message": "File uploaded successfully.",
            "filename": filename 
        }), 200
        
    except Exception as e:
        logging.error(f"Error in upload endpoint: {e}")
        return jsonify({
            "success": False,
            "error": f"Server error during upload: {str(e)}"
        }), 500


# ----------------------------------------------------------------------
# >>> STEP 2: CONVERT ROUTE (Processes file, deletes file) <<<
# ----------------------------------------------------------------------
@app.route('/convert', methods=['POST'])
def convert_file_from_upload():
    """
    Handle PDF conversion by file path (the filename from the /upload step).
    """
    if not request.is_json:
        return jsonify({
            "success": False, 
            "error": "Content-Type must be application/json"
        }), 400
    
    data = request.get_json()
    
    # pdf_path is now the filename returned from /upload
    filename = data.get('pdf_path') 
    
    if not filename or filename.strip() == '':
        return jsonify({
            "success": False, 
            "error": "Missing 'pdf_path' (filename from /upload) parameter in JSON body"
        }), 400
    
    file_to_delete = UPLOAD_FOLDER / filename
    
    try:
        max_pages = int(data.get('max_pages', 20))
        logging.info(f"Converting file from upload directory: {filename}")
        
        # Convert PDF: The function now handles path resolution
        result = convert_PDF(filename, max_pages=max_pages)
        
        return jsonify(result)
        
    except FileNotFoundError as e:
        # File was not found in the upload directory
        return jsonify({
            "success": False,
            "error": str(e)
        }), 404
    except RuntimeError as e:
        # OCR processing failed
        return jsonify({
            "success": False, 
            "error": str(e)
        }), 500
    except Exception as e:
        logging.error(f"Error in convert endpoint: {e}")
        return jsonify({
            "success": False, 
            "error": f"Server error: {str(e)}"
        }), 500
    finally:
        # CRITICAL: Clean up the file after processing
        if file_to_delete.exists():
            try:
                os.remove(file_to_delete)
                logging.info(f"Cleaned up processed file: {file_to_delete}")
            except Exception as e:
                logging.warning(f"Failed to remove processed file: {e}")


if __name__ == "__main__":
    logging.info("Starting Flask PDF OCR API")
    use_GPU = torch.cuda.is_available()
    logging.info(f"Using GPU: {use_GPU}")
    logging.info(f"Upload folder: {UPLOAD_FOLDER}")
    logging.info("Flask API ready")
    logging.info("Access API at: http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)
