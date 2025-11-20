import logging
import time
from pathlib import Path
import contextlib
import os

import requests  # <-- for downloading from download_link
from werkzeug.utils import secure_filename  # Used for safe filename handling

from flask import Flask, request, jsonify
import nltk
import torch
import base64
import uuid
import binascii
# Assuming pdf2text.py is in the same directory and contains
# rm_local_text_files, convert_PDF_to_Text, and ocr_predictor
from pdf2text import *

# --- CONFIGURATION ---
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
            assume_straight_pages=False,
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
        raise FileNotFoundError(f"File '{filename}' not found on server. Did you call /upload or /api/upload-file first?")
    
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
                "description": "Step 1 (multipart): Uploads file to server, returns filename for /convert",
            },
            "/api/upload-file": {
                "method": "POST",
                "description": "Step 1 (OpenAI JSON): Accepts openaiFileIdRefs, downloads file, saves it locally, returns filename for /convert",
            },
            "/convert": {
                "method": "POST",
                "description": "Step 2: Converts the file saved by /upload or /api/upload-file, returns text, and deletes the file.",
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
# >>> STEP 1A: UPLOAD ROUTE (Saves file from multipart/form-data) <<<
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
# >>> STEP 1B: OPENAI JSON ROUTE (openaiFileIdRefs -> local file) <<<
# ----------------------------------------------------------------------
@app.route('/api/upload-file', methods=['POST'])
def upload_file_from_openai():
    """
    Handles JSON payloads from the OpenAI Assistant that contain openaiFileIdRefs.
    
    Flow:
    - Reads openaiFileIdRefs from JSON
    - Downloads the file from the temporary link
    - Validates extension
    - Saves it into UPLOAD_FOLDER (transient local storage)
    - Returns the filename so /convert can process it
    """
    data = request.get_json(silent=True) or {}
    refs = data.get('openaiFileIdRefs', [])

    if not refs:
        return jsonify({
            "success": False,
            "error": "No openaiFileIdRefs in JSON request",
            "received_content_type": request.content_type
        }), 400

    # Original openaiFileIdRefs logic
    first = refs[0]
    if isinstance(first, dict):
        download_link = first.get('download_link')
        original_name = first.get('name') or 'uploaded_file'
    else:
        download_link = first
        original_name = 'uploaded_file'

    if not download_link:
        return jsonify({
            "success": False,
            "error": "Missing download_link in openaiFileIdRefs"
        }), 400

    try:
        # 1. Download the file from the temporary link
        app.logger.info(f"Downloading file from: {download_link}")
        r = requests.get(download_link, stream=False, timeout=10)
        r.raise_for_status()
        file_content_bytes = r.content

        # 2. Validate and sanitize filename
        filename_safe = secure_filename(original_name) or "uploaded_file"
        if '.' not in filename_safe:
            return jsonify({
                "success": False,
                "error": "Uploaded file has no extension; cannot determine type"
            }), 400

        if not allowed_file(filename_safe):
            return jsonify({
                "success": False,
                "error": f"File type not allowed. Allowed types: {', '.join(ALLOWED_EXTENSIONS)}"
            }), 400

        # Extract extension and make filename unique
        base_name, ext = filename_safe.rsplit('.', 1)
        ext = ext.lower()
        unique_filename = f"{base_name}_{int(time.time())}.{ext}"
        file_path = UPLOAD_FOLDER / unique_filename

        # 3. Save the downloaded file to disk (transient)
        with open(file_path, "wb") as f:
            f.write(file_content_bytes)

        app.logger.info(f"File downloaded from OpenAI and saved to: {file_path}")

        # 4. Return filename (to be used as pdf_path in /convert)
        return jsonify({
            "success": True,
            "message": "File downloaded and saved successfully from openaiFileIdRefs.",
            "filename": unique_filename
        }), 200

    except requests.RequestException as e:
        app.logger.error(f"Request failed (download): {e}", exc_info=True)
        return jsonify({
            "success": False,
            "error": f"Failed to download file from temporary link: {str(e)}"
        }), 502
    except Exception as e:
        app.logger.error(f"General error in /api/upload-file: {e}", exc_info=True)
        return jsonify({
            "success": False,
            "error": f"Server error during OpenAI-style upload: {str(e)}"
        }), 500


# ----------------------------------------------------------------------
# >>> STEP 2: CONVERT ROUTE (Processes file, deletes file) <<<
# ----------------------------------------------------------------------
@app.route('/convert', methods=['POST'])
def convert_file_from_upload():
    """
    Handle PDF conversion by file path (the filename from the /upload or /api/upload-file step).
    """
    if not request.is_json:
        return jsonify({
            "success": False, 
            "error": "Content-Type must be application/json"
        }), 400
    
    data = request.get_json()
    
    # pdf_path is now the filename returned from /upload or /api/upload-file
    filename = data.get('pdf_path') 
    
    if not filename or filename.strip() == '':
        return jsonify({
            "success": False, 
            "error": "Missing 'pdf_path' (filename from /upload or /api/upload-file) parameter in JSON body"
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
        # CRITICAL: Clean up the file after processing (no long-term storage)
        if file_to_delete.exists():
            try:
                os.remove(file_to_delete)
                logging.info(f"Cleaned up processed file: {file_to_delete}")
            except Exception as e:
                logging.warning(f"Failed to remove processed file: {e}")




@app.route('/process-base64', methods=['POST'])
def process_base64():
    # 1. VALIDATION: Check for JSON
    if not request.is_json:
        return jsonify({"success": False, "error": "Content-Type must be application/json"}), 400
    
    data = request.get_json()
    
    # 2. EXTRACTION: Get your specific keys
    input_filename = data.get('file_name')             # e.g., "Valeo.csv" (or .pdf)
    base64_string = data.get('file_content_base64')    # The data string
    
    if not input_filename or not base64_string:
        return jsonify({
            "success": False, 
            "error": "Missing 'file_name' or 'file_content_base64' in payload"
        }), 400

    # 3. FILENAME SETUP
    # Extract the extension (e.g., .pdf, .png) from the user's filename
    _, ext = os.path.splitext(input_filename)
    
    # Create a unique name to avoid collisions: "uuid_Valeo.pdf"
    # We use secure_filename to remove spaces/slashes from the user input
    safe_name = secure_filename(input_filename)
    unique_filename = f"{uuid.uuid4()}_{safe_name}"
    
    temp_file_path = UPLOAD_FOLDER / unique_filename

    # Optional: Check if extension is allowed (based on your global ALLOWED_EXTENSIONS)
    # if ext.lower().replace('.', '') not in ALLOWED_EXTENSIONS:
    #     return jsonify({"success": False, "error": "File type not allowed"}), 400

    try:
        # 4. DECODE & SAVE
        if "," in base64_string:
            base64_string = base64_string.split(",", 1)[1]

        file_content = base64.b64decode(base64_string)
        
        with open(temp_file_path, "wb") as f:
            f.write(file_content)
        
        logging.info(f"Temp file saved: {unique_filename}")

        # 5. PROCESS (OCR)
        # Note: If you send a .csv, ensure convert_PDF can handle it. 
        # Usually this function expects .pdf or images.
        max_pages = int(data.get('max_pages', 20))
        
        result = convert_PDF(unique_filename, max_pages=max_pages)
        
        return jsonify(result)

    except (binascii.Error, ValueError):
        return jsonify({"success": False, "error": "Invalid Base64 string"}), 400
    except Exception as e:
        logging.error(f"Error in base64 processing: {e}")
        return jsonify({"success": False, "error": str(e)}), 500
        
    finally:
        # 6. CLEANUP
        if temp_file_path.exists():
            try:
                os.remove(temp_file_path)
                logging.info(f"Cleaned up: {unique_filename}")
            except Exception as e:
                logging.warning(f"Failed to delete temp file: {e}")





if __name__ == "__main__":
    logging.info("Starting Flask PDF OCR API")
    use_GPU = torch.cuda.is_available()
    logging.info(f"Using GPU: {use_GPU}")
    logging.info(f"Upload folder: {UPLOAD_FOLDER}")
    logging.info("Flask API ready")
    logging.info("Access API at: http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)
