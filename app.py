import logging
import time
from pathlib import Path
import contextlib
import os
from datetime import datetime, timedelta
import uuid
import binascii

import requests
from werkzeug.utils import secure_filename

from flask import Flask, request, jsonify, send_file
import nltk
import torch
import base64

# Assuming pdf2text.py is in the same directory
from pdf2text import *
from pdf2image import convert_from_path
import io
from PIL import Image
import urllib.parse

# --- CONFIGURATION ---
UPLOAD_FOLDER = Path('/tmp/uploads')
OUTPUT_FOLDER = Path('/tmp/output_images')  # Store converted images here
ALLOWED_EXTENSIONS = {'pdf', 'png', 'jpg', 'jpeg', 'tiff', 'bmp'}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

app = Flask(__name__)

# Ensure directories exist
UPLOAD_FOLDER.mkdir(exist_ok=True, parents=True)
OUTPUT_FOLDER.mkdir(exist_ok=True, parents=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
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


def cleanup_old_files(folder, max_age_hours=24):
    """Remove files older than max_age_hours from folder"""
    try:
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        for file_path in folder.iterdir():
            if file_path.is_file():
                file_time = datetime.fromtimestamp(file_path.stat().st_mtime)
                if file_time < cutoff_time:
                    os.remove(file_path)
                    logging.info(f"Cleaned up old file: {file_path}")
    except Exception as e:
        logging.warning(f"Cleanup failed: {e}")


# --- REFACTORED CONVERSION CORE (OCR) ---
def convert_PDF(filename: str, max_pages=20):
    """
    Core function to convert a file located in UPLOAD_FOLDER to text using OCR.
    
    Args:
        filename (str): The name of the file saved in the UPLOAD_FOLDER.
        max_pages (int): Maximum number of pages to process.
    """
    rm_local_text_files()
    global ocr_model
    
    st = time.perf_counter()
    file_path = UPLOAD_FOLDER / filename
    
    if not file_path.exists():
        logging.error(f"File {file_path} does not exist")
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
        raise RuntimeError(f"OCR processing failed: {str(e)}")


# --- INFO & HEALTH ENDPOINTS ---
@app.route('/')
def index():
    """API info endpoint"""
    return jsonify({
        "message": "PDF OCR API with Image URL Support",
        "version": "5.1",
        "endpoints": {
            "/upload": {
                "method": "POST",
                "description": "Step 1: Upload file, returns filename",
            },
            "/api/upload-file": {
                "method": "POST",
                "description": "Step 1 (OpenAI): Accepts openaiFileIdRefs, returns filename",
            },
            "/convert": {
                "method": "POST",
                "description": "Step 2: OCR conversion to text",
            },
            "/convert-to-images": {
                "method": "POST",
                "description": "Step 2 (New): Convert PDF to image URLs for GPT (with wait time)",
            },
            "/images/<filename>": {
                "method": "GET",
                "description": "Serve converted images to GPT Assistant",
            },
            "/process-base64": {
                "method": "POST",
                "description": "Process base64 encoded files",
            },
            "/cleanup": {
                "method": "POST",
                "description": "Manually trigger file cleanup",
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
        "output_folder": str(OUTPUT_FOLDER),
        "upload_folder_exists": UPLOAD_FOLDER.exists(),
        "output_folder_exists": OUTPUT_FOLDER.exists()
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
        base_name, ext = filename.rsplit('.', 1)
        unique_filename = f"{base_name}_{int(time.time())}.{ext}"
        
        filepath = UPLOAD_FOLDER / unique_filename
        file.save(str(filepath)) 
        logging.info(f"File successfully uploaded and saved to: {filepath}")
        
        return jsonify({
            "success": True,
            "message": "File uploaded successfully.",
            "filename": unique_filename 
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
    """
    data = request.get_json(silent=True) or {}
    refs = data.get('openaiFileIdRefs', [])

    if not refs:
        return jsonify({
            "success": False,
            "error": "No openaiFileIdRefs in JSON request",
            "received_content_type": request.content_type
        }), 400

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
        app.logger.info(f"Downloading file from: {download_link}")
        r = requests.get(download_link, stream=False, timeout=10)
        r.raise_for_status()
        file_content_bytes = r.content

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

        base_name, ext = filename_safe.rsplit('.', 1)
        ext = ext.lower()
        unique_filename = f"{base_name}_{int(time.time())}.{ext}"
        file_path = UPLOAD_FOLDER / unique_filename

        with open(file_path, "wb") as f:
            f.write(file_content_bytes)

        app.logger.info(f"File downloaded from OpenAI and saved to: {file_path}")

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
# >>> STEP 2A: CONVERT ROUTE (OCR - Processes file to text) AVEC DÉLAI <<<
# ----------------------------------------------------------------------
@app.route('/convert', methods=['POST'])
def convert_file_from_upload():
    """
    Handle PDF conversion by file path (the filename from the /upload or /api/upload-file step).
    Converts to TEXT using OCR.
    Attend quelques secondes pour s'assurer que le fichier est complètement uploadé.
    """
    if not request.is_json:
        return jsonify({
            "success": False, 
            "error": "Content-Type must be application/json"
        }), 400
    
    data = request.get_json()
    filename = data.get('pdf_path') 
    
    if not filename or filename.strip() == '':
        return jsonify({
            "success": False, 
            "error": "Missing 'pdf_path' (filename from /upload or /api/upload-file) parameter in JSON body"
        }), 400
    
    # Configuration du délai d'attente
    wait_time = int(data.get('wait_time', 3))  # 3 secondes par défaut
    
    file_to_delete = UPLOAD_FOLDER / filename
    
    # ⏱️ ATTENDRE que le fichier soit complètement uploadé
    logging.info(f"Waiting {wait_time} seconds for file to be fully uploaded...")
    time.sleep(wait_time)
    
    # Vérifier si le fichier existe après l'attente
    if not file_to_delete.exists():
        # Attendre encore un peu et réessayer
        logging.warning(f"File not found after {wait_time}s, waiting 2 more seconds...")
        time.sleep(2)
        
        if not file_to_delete.exists():
            return jsonify({
                "success": False,
                "error": f"File '{filename}' not found even after waiting {wait_time + 2} seconds"
            }), 404
    
    # Vérifier que le fichier n'est pas vide
    file_size = file_to_delete.stat().st_size
    if file_size == 0:
        return jsonify({
            "success": False,
            "error": "File is empty or still being written"
        }), 400
    
    logging.info(f"File found and ready: {filename} ({file_size} bytes)")
    
    try:
        max_pages = int(data.get('max_pages', 20))
        logging.info(f"Converting file from upload directory: {filename}")
        
        result = convert_PDF(filename, max_pages=max_pages)
        
        # Ajouter le wait_time utilisé dans la réponse
        result['wait_time_used'] = wait_time
        
        return jsonify(result)
        
    except FileNotFoundError as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 404
    except RuntimeError as e:
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
        if file_to_delete.exists():
            try:
                os.remove(file_to_delete)
                logging.info(f"Cleaned up processed file: {file_to_delete}")
            except Exception as e:
                logging.warning(f"Failed to remove processed file: {e}")


# ----------------------------------------------------------------------
# >>> STEP 2B: CONVERT TO IMAGES (AVEC DÉLAI D'ATTENTE) <<<
# ----------------------------------------------------------------------
@app.route('/convert-to-images', methods=['POST'])
def convert_pdf_to_images():
    """
    Convert PDF to JPEG images, save them to OUTPUT_FOLDER,
    and return URLs that the GPT Assistant can access.
    Attend quelques secondes pour s'assurer que le fichier est complètement uploadé.
    """
    if not request.is_json:
        return jsonify({
            "success": False,
            "error": "Content-Type must be application/json"
        }), 400
    
    data = request.get_json()
    filename = data.get('pdf_path')
    
    if not filename or filename.strip() == '':
        return jsonify({
            "success": False,
            "error": "Missing 'pdf_path' parameter"
        }), 400
    
    # Configuration
    dpi = int(data.get('dpi', 200))
    quality = int(data.get('quality', 85))
    max_pages = int(data.get('max_pages', 50))
    wait_time = int(data.get('wait_time', 3))  # Délai configurable (3 secondes par défaut)
    
    pdf_path = UPLOAD_FOLDER / filename
    
    # ⏱️ ATTENDRE que le fichier soit complètement uploadé
    logging.info(f"Waiting {wait_time} seconds for file to be fully uploaded...")
    time.sleep(wait_time)
    
    # Vérifier si le fichier existe après l'attente
    if not pdf_path.exists():
        # Attendre encore un peu et réessayer
        logging.warning(f"File not found after {wait_time}s, waiting 2 more seconds...")
        time.sleep(2)
        
        if not pdf_path.exists():
            return jsonify({
                "success": False,
                "error": f"File '{filename}' not found even after waiting {wait_time + 2} seconds"
            }), 404
    
    # Vérifier que le fichier n'est pas vide et est complètement écrit
    file_size = pdf_path.stat().st_size
    if file_size == 0:
        return jsonify({
            "success": False,
            "error": "File is empty or still being written"
        }), 400
    
    logging.info(f"File found and ready: {filename} ({file_size} bytes)")
    
    try:
        # Cleanup old files first
        cleanup_old_files(OUTPUT_FOLDER, max_age_hours=24)
        
        # Convert PDF to images
        logging.info(f"Converting PDF to images: {filename}")
        pages = convert_from_path(pdf_path, dpi=dpi)
        
        # Limit pages
        total_pages = len(pages)
        if len(pages) > max_pages:
            pages = pages[:max_pages]
            truncated = True
        else:
            truncated = False
        
        # Save each page as JPEG
        image_urls = []
        base_name = filename.rsplit('.', 1)[0]
        timestamp = int(time.time())
        
        for i, page in enumerate(pages):
            # Create unique filename for each page
            image_filename = f"{base_name}_page_{i+1}_{timestamp}.jpg"
            image_path = OUTPUT_FOLDER / image_filename
            
            # Optimize and save
            page.save(
                str(image_path), 
                'JPEG', 
                quality=quality, 
                optimize=True
            )
            
            # Create URL that GPT can access
            base_url = request.host_url.rstrip('/')
            image_url = f"{base_url}/images/{image_filename}"
            
            image_urls.append({
                "page": i + 1,
                "url": image_url,
                "filename": image_filename
            })
            
            logging.info(f"Saved page {i+1}: {image_filename}")
        
        # Clean up original PDF
        os.remove(pdf_path)
        logging.info(f"Cleaned up PDF: {pdf_path}")
        
        return jsonify({
            "success": True,
            "message": "PDF converted to images successfully",
            "total_pages": total_pages,
            "converted_pages": len(image_urls),
            "truncated": truncated,
            "wait_time_used": wait_time,
            "images": image_urls,
            "note": "Images will be automatically deleted after 24 hours"
        }), 200
        
    except Exception as e:
        logging.error(f"Conversion error: {e}")
        return jsonify({
            "success": False,
            "error": f"Conversion failed: {str(e)}"
        }), 500


# ----------------------------------------------------------------------
# >>> STEP 3: SERVE IMAGES (GPT Assistant accesses these URLs) <<<
# ----------------------------------------------------------------------
@app.route('/images/<filename>', methods=['GET'])
def serve_image(filename):
    """
    Serve the converted image files so GPT Assistant can access them.
    This is the URL that will be returned in the conversion response.
    """
    try:
        # Secure the filename to prevent directory traversal
        safe_filename = secure_filename(filename)
        image_path = OUTPUT_FOLDER / safe_filename
        
        if not image_path.exists():
            return jsonify({
                "success": False,
                "error": "Image not found"
            }), 404
        
        # Send the image file
        return send_file(
            str(image_path),
            mimetype='image/jpeg',
            as_attachment=False,
            download_name=safe_filename
        )
        
    except Exception as e:
        logging.error(f"Error serving image: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


# ----------------------------------------------------------------------
# >>> BASE64 PROCESSING ROUTE <<<
# ----------------------------------------------------------------------
@app.route('/process-base64', methods=['POST'])
def process_base64():
    """Process base64 encoded files"""
    if not request.is_json:
        return jsonify({"success": False, "error": "Content-Type must be application/json"}), 400
    
    data = request.get_json()
    
    input_filename = data.get('file_name')
    base64_string = data.get('file_content_base64')
    
    if not input_filename or not base64_string:
        return jsonify({
            "success": False, 
            "error": "Missing 'file_name' or 'file_content_base64' in payload"
        }), 400

    _, ext = os.path.splitext(input_filename)
    safe_name = secure_filename(input_filename)
    unique_filename = f"{uuid.uuid4()}_{safe_name}"
    
    temp_file_path = UPLOAD_FOLDER / unique_filename

    try:
        if "," in base64_string:
            base64_string = base64_string.split(",", 1)[1]

        file_content = base64.b64decode(base64_string)
        
        with open(temp_file_path, "wb") as f:
            f.write(file_content)
        
        logging.info(f"Temp file saved: {unique_filename}")

        max_pages = int(data.get('max_pages', 20))
        
        result = convert_PDF(unique_filename, max_pages=max_pages)
        
        return jsonify(result)

    except (binascii.Error, ValueError):
        return jsonify({"success": False, "error": "Invalid Base64 string"}), 400
    except Exception as e:
        logging.error(f"Error in base64 processing: {e}")
        return jsonify({"success": False, "error": str(e)}), 500
        
    finally:
        if temp_file_path.exists():
            try:
                os.remove(temp_file_path)
                logging.info(f"Cleaned up: {unique_filename}")
            except Exception as e:
                logging.warning(f"Failed to delete temp file: {e}")


# ----------------------------------------------------------------------
# >>> CLEANUP ENDPOINT <<<
# ----------------------------------------------------------------------
@app.route('/cleanup', methods=['POST'])
def manual_cleanup():
    """
    Manually trigger cleanup of old files.
    Useful for maintenance.
    """
    try:
        max_age = int(request.get_json(silent=True).get('max_age_hours', 1)) if request.is_json else 1
        
        cleanup_old_files(OUTPUT_FOLDER, max_age_hours=max_age)
        cleanup_old_files(UPLOAD_FOLDER, max_age_hours=max_age)
        
        return jsonify({
            "success": True,
            "message": f"Cleanup completed (files older than {max_age} hours removed)"
        }), 200
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

#------------------------------------------------------------------------------------------------------------------------------------------------------------


# --- CONFIGURATION (No Token Needed) ---
GITHUB_OWNER = "STS-Engineer" 
GITHUB_REPO = "RFQ-back"
GITHUB_BRANCH = "main" # or 'master', check your repo

@app.route('/process-rfq-from-github', methods=['POST'])
def process_rfq_github():
    if not request.is_json:
        return jsonify({"success": False, "error": "Content-Type must be application/json"}), 400

    data = request.get_json()
    rfq_path = data.get('rfq_file_path')

    if not rfq_path:
        return jsonify({"success": False, "error": "Missing 'rfq_file_path'"}), 400

    local_file_path = None

    try:
        # --- STEP 1: CONSTRUCT PUBLIC RAW URL ---
        # Format: https://raw.githubusercontent.com/{USER}/{REPO}/{BRANCH}/{PATH}
        
        # Clean the path (remove leading slash)
        clean_path = rfq_path.strip("/")
        
        # Handle spaces in filenames (e.g., "my file.pdf" -> "my%20file.pdf")
        # We use requests params or simple replacement for safety
        import urllib.parse
        encoded_path = urllib.parse.quote(clean_path)
        
        url = f"https://raw.githubusercontent.com/{GITHUB_OWNER}/{GITHUB_REPO}/{GITHUB_BRANCH}/{encoded_path}"
        
        logging.info(f"Downloading from Public GitHub: {url}")
        
        # Download (No headers/auth needed for public raw files)
        response = requests.get(url, stream=True)

        if response.status_code == 404:
            return jsonify({"success": False, "error": f"File not found on GitHub. Check path: {clean_path}"}), 404
        elif response.status_code != 200:
            return jsonify({"success": False, "error": f"Download failed: {response.status_code}"}), 400

        # --- STEP 2: SAVE LOCALLY (Simulate Upload) ---
        original_filename = os.path.basename(clean_path)
        # Decode the filename back to normal text if it was encoded
        original_filename = urllib.parse.unquote(original_filename)
        
        safe_filename = secure_filename(original_filename)
        base_name, ext = os.path.splitext(safe_filename)
        unique_filename = f"{base_name}_{int(time.time())}{ext}"
        local_file_path = UPLOAD_FOLDER / unique_filename

        with open(local_file_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        logging.info(f"File saved locally: {local_file_path}")

        # --- STEP 3: CONVERT ---
        max_pages = int(data.get('max_pages', 20))
        result = convert_PDF(unique_filename, max_pages=max_pages)
        result['source_rfq_path'] = rfq_path
        
        return jsonify(result)

    except Exception as e:
        logging.error(f"Error in GitHub processing: {e}")
        return jsonify({"success": False, "error": str(e)}), 500
        
    finally:
        # Cleanup
        if local_file_path and local_file_path.exists():
            try:
                os.remove(local_file_path)
            except Exception as e:
                logging.warning(f"Failed to cleanup: {e}")










if __name__ == "__main__":
    logging.info("Starting Flask PDF OCR API with Image URL Support")
    use_GPU = torch.cuda.is_available()
    logging.info(f"Using GPU: {use_GPU}")
    logging.info(f"Upload folder: {UPLOAD_FOLDER}")
    logging.info(f"Output folder: {OUTPUT_FOLDER}")
    logging.info("Flask API ready")
    logging.info("Access API at: http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)
