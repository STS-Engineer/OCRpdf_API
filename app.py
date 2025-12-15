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
from werkzeug.middleware.proxy_fix import ProxyFix
from flask import Flask, request, jsonify, send_file, send_from_directory
import nltk
import torch
import base64

# --- PADDLEOCR RELATED IMPORTS ---
from paddleocr import PaddleOCR
import cv2
import numpy as np
from typing import List, Dict, Any, Optional
# ---------------------------------

# Assuming pdf2text.py is in the same directory
from pdf2text import *
from pdf2image import convert_from_path
import io
from PIL import Image
import urllib.parse
import psycopg2
import fitz
# --- CONFIGURATION ---
UPLOAD_FOLDER = Path('/tmp/uploads')
OUTPUT_FOLDER = Path('/tmp/output_images')  # Store converted images here
ALLOWED_EXTENSIONS = {'pdf', 'png', 'jpg', 'jpeg', 'tiff', 'bmp'}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

app = Flask(__name__)
# Ceci indique à Flask de faire confiance aux en-têtes du proxy Azure (X-Forwarded-Proto)
app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)
# Ensure directories exist
UPLOAD_FOLDER.mkdir(exist_ok=True, parents=True)
OUTPUT_FOLDER.mkdir(exist_ok=True, parents=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024

# Initialize NLTK data
nltk.download("stopwords")

# Initialize docTR OCR model globally
ocr_model = None


def init_ocr_model():
    """Initialize the docTR OCR model"""
    global ocr_model
    logging.info("Loading docTR OCR model...")
    with contextlib.redirect_stdout(None):
        ocr_model = ocr_predictor(
            "db_resnet50",
            "crnn_mobilenet_v3_large",
            pretrained=True,
            assume_straight_pages=False,
        )
    logging.info("docTR OCR model loaded successfully into the application process.")


# Pre-load docTR model at module level for Gunicorn
try:
    init_ocr_model()
except Exception as e:
    logging.error(f"FATAL: Failed to load docTR OCR model during Gunicorn preload: {e}", exc_info=True)


# --- PADDLEOCR INITIALIZATION (MODIFIED FOR MULTILANGUAGE) ---
paddle_ocr_model = None

def init_paddle_ocr():
    """Initialize PaddleOCR model for Chinese, English, and German support."""
    global paddle_ocr_model
    logging.info("Loading PaddleOCR model (lang=ch for Chinese/English/Latin support)...")
    try:
        # 'ch' is the standard abbreviation for the Chinese & English model in PaddleOCR.
        # This model is generally robust for mixed Chinese and Latin (English, German, etc.) scripts.
        # If the document was primarily German, 'lang="german"' or 'lang="de"' (depending on version)
        # would be used, but 'ch' is the best for Chinese+Latin mixed content.
        paddle_ocr_model = PaddleOCR(
            lang="ch", 
            use_angle_cls=False,
            use_gpu=torch.cuda.is_available()
        )
        logging.info("PaddleOCR model loaded successfully (lang='ch').")
    except Exception as e:
        logging.error(f"Failed to load PaddleOCR model: {e}", exc_info=True)

# Pre-load PaddleOCR at module level
try:
    init_paddle_ocr()
except Exception as e:
    logging.error(f"FATAL: Failed to load PaddleOCR during startup: {e}", exc_info=True)


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
    Core function to convert a file located in UPLOAD_FOLDER to text using docTR OCR.
    
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


# --- PADDLEOCR HELPER FUNCTIONS ---

def prepare_image_for_paddle(img: np.ndarray, target_max_side: int = 2200) -> tuple:
    """
    Upscale image if needed so largest side is target_max_side.
    Returns (processed_image, scale_factor)
    """
    h, w = img.shape[:2]
    max_side = max(h, w)
    
    if max_side < target_max_side:
        scale = target_max_side / max_side
        new_w = int(w * scale)
        new_h = int(h * scale)
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        logging.info(f"Upscaled image from {w}x{h} to {new_w}x{new_h}")
        return resized, scale
    else:
        logging.info(f"Using original resolution {w}x{h}")
        return img, 1.0


def normalize_poly(poly: Any, scale: float = 1.0) -> List[List[int]]:
    """Convert PaddleOCR polygon to list of [x,y] coordinates"""
    pts: List[List[int]] = []
    if poly is None:
        return pts
    
    try:
        arr = np.array(poly)
    except Exception:
        return pts
    
    if arr.ndim == 2 and arr.shape[1] >= 2:
        for x, y in arr[:, :2]:
            try:
                x = float(x) / scale
                y = float(y) / scale
                pts.append([int(round(x)), int(round(y))])
            except (TypeError, ValueError):
                continue
    
    return pts


def run_paddle_ocr_on_image(image: np.ndarray) -> List[Dict[str, Any]]:
    """
    Run PaddleOCR on an image array.
    
    Returns list of:
    [
      {
        "text": str,
        "confidence": float,
        "box": [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
      },
      ...
    ]
    """
    global paddle_ocr_model
    
    if paddle_ocr_model is None:
        raise RuntimeError("PaddleOCR model not initialized")
    
    # Prepare image
    processed_img, scale = prepare_image_for_paddle(image)
    
    # Save to temporary file (PaddleOCR requires file path)
    temp_path = UPLOAD_FOLDER / f"temp_paddle_{int(time.time())}.png"
    cv2.imwrite(str(temp_path), processed_img)
    
    try:
        # Run OCR
        result = paddle_ocr_model.ocr(str(temp_path), cls=False)
        
        entries: List[Dict[str, Any]] = []
        
        if not result or not result[0]:
            logging.warning("PaddleOCR returned empty result")
            return entries
        
        # Parse results
        for line in result[0]:
            if not line:
                continue
            
            box = line[0]  # Bounding box coordinates
            text_info = line[1]  # (text, confidence)
            
            text = text_info[0]
            confidence = float(text_info[1])
            
            if not text:
                continue
            
            # Normalize box coordinates back to original scale
            box_pts = normalize_poly(box, scale=scale)
            
            if len(box_pts) < 3:
                continue
            
            entries.append({
                "text": str(text),
                "confidence": confidence,
                "box": box_pts
            })
        
        logging.info(f"PaddleOCR found {len(entries)} text boxes")
        return entries
        
    except Exception as e:
        logging.error(f"PaddleOCR processing failed: {e}", exc_info=True)
        raise
    finally:
        # Cleanup temp file
        if temp_path.exists():
            try:
                os.remove(temp_path)
            except Exception as e:
                logging.warning(f"Failed to remove temp file: {e}")


def save_annotated_paddle_image(image: np.ndarray, 
                                 entries: List[Dict[str, Any]],
                                 output_filename: str) -> str:
    """
    Draw OCR boxes on image and save to OUTPUT_FOLDER.
    Returns the filename (not full URL).
    """
    annotated = image.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 2
    
    for idx, entry in enumerate(entries, start=1):
        box = entry.get("box")
        text = entry.get("text", "")
        
        if not box or len(box) < 3:
            continue
        
        pts = np.array(box, dtype=np.int32).reshape((-1, 1, 2))
        
        # Draw green polygon
        cv2.polylines(annotated, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
        
        # Draw number and text near first point
        x, y = box[0]
        label = f"{idx}"
        
        # Draw index number
        cv2.putText(
            annotated,
            label,
            (x, y - 10),
            font,
            font_scale,
            (0, 0, 255),
            thickness,
            lineType=cv2.LINE_AA
        )
        
        # Draw text below the box (truncate if too long)
        display_text = text[:20] + "..." if len(text) > 20 else text
        cv2.putText(
            annotated,
            display_text,
            (x, y + 20),
            font,
            font_scale * 0.7,
            (255, 0, 0),
            1,
            lineType=cv2.LINE_AA
        )
    
    # Save annotated image
    output_path = OUTPUT_FOLDER / output_filename
    cv2.imwrite(str(output_path), annotated)
    logging.info(f"Saved annotated image to: {output_path}")
    
    return output_filename


# --- INFO & HEALTH ENDPOINTS ---
@app.route('/')
def index():
    """API info endpoint"""
    return jsonify({
        "message": "PDF OCR API with Image URL Support",
        "version": "5.2 (Multilang OCR)",
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
                "description": "Step 2: OCR conversion to text (docTR)",
            },
            "/convert-to-images": {
                "method": "POST",
                "description": "Step 2 (New): Convert PDF to image URLs for GPT (with wait time)",
            },
            "/images/<filename>": {
                "method": "GET",
                "description": "Serve converted images to GPT Assistant",
            },
            "/download-image/<filename>": {
                "method": "GET",
                "description": "Force download of converted images",
            },
            "/process-base64": {
                "method": "POST",
                "description": "Process base64 encoded files",
            },
            "/process-rfq-id-to-images-with-ocr": {
                "method": "POST",
                "description": "NEW: Process RFQ ID to images with PaddleOCR for detailed results/boxes",
            },
            "/process-rfq-id-ocr-only": {
                "method": "POST",
                "description": "NEW: Process RFQ ID to text only using PaddleOCR (lightweight)",
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
        "docTR_model_loaded": ocr_model is not None,  
        "paddle_ocr_model_loaded": paddle_ocr_model is not None, 
        "paddle_ocr_language": "ch (Chinese, English, German/Latin script support)", # Updated
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
    Converts to TEXT using docTR OCR.
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
        return jsonify({"success": False, "error": "Content-Type must be application/json"}), 400
    
    data = request.get_json()
    filename = data.get('pdf_path')
    
    if not filename or filename.strip() == '':
        return jsonify({"success": False, "error": "Missing 'pdf_path' parameter"}), 400
    
    # Configuration
    dpi = int(data.get('dpi', 600))
    quality = int(data.get('quality', 95))
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
        return jsonify({"success": False, "error": "File is empty or still being written"}), 400
    
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
    """Serve image for viewing in browser"""
    try:
        safe_filename = secure_filename(filename)
        image_path = OUTPUT_FOLDER / safe_filename
        
        if not image_path.exists():
            return jsonify({"success": False, "error": "Image not found"}), 404
        
        # Determine MIME type based on extension
        mime_type = 'image/png' if filename.lower().endswith('.png') else 'image/jpeg'
        
        return send_file(
            str(image_path),
            mimetype=mime_type,
            as_attachment=False,
            download_name=safe_filename
        )
    except Exception as e:
        logging.error(f"Error serving image: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/download-image/<filename>', methods=['GET'])
def download_image(filename):
    """Force download of image file"""
    try:
        safe_filename = secure_filename(filename)
        image_path = OUTPUT_FOLDER / safe_filename
        
        if not image_path.exists():
            return jsonify({"success": False, "error": "Image not found"}), 404
        
        # Determine MIME type based on extension
        if filename.lower().endswith('.png'):
            mime_type = 'image/png'
        elif filename.lower().endswith('.jpg') or filename.lower().endswith('.jpeg'):
            mime_type = 'image/jpeg'
        else:
            mime_type = 'application/octet-stream'
        
        return send_file(
            str(image_path),
            mimetype=mime_type,
            as_attachment=True,  # This forces download
            download_name=safe_filename
        )
    except Exception as e:
        logging.error(f"Error downloading image: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


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

DB_CONFIG = {
    "host": "avo-adb-002.postgres.database.azure.com",
    "database": "RFQ_DATA",
    "user": "administrationSTS",
    "password": "St$@0987"
}


@app.route('/process-rfq-id', methods=['POST'])
def process_rfq_by_id():
    """
    1. Accepts 'rfq_id' in JSON.
    2. Queries Postgres to get 'rfq_file_path'.
    3. Downloads file from Public GitHub.
    4. Runs docTR OCR.
    """
    if not request.is_json:
        return jsonify({"success": False, "error": "Content-Type must be application/json"}), 400

    data = request.get_json()
    rfq_id = data.get('rfq_id')

    if not rfq_id:
        return jsonify({"success": False, "error": "Missing 'rfq_id'"}), 400

    local_file_path = None
    conn = None
    cur = None

    try:
        # --- STEP 1: FETCH PATH FROM DB ---
        logging.info(f"Connecting to DB to fetch path for RFQ ID: {rfq_id}")
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()
        
        # Query public.main for the path
        query = "SELECT rfq_file_path FROM public.main WHERE rfq_id = %s"
        cur.execute(query, (rfq_id,))
        result = cur.fetchone()
        
        if not result:
            return jsonify({"success": False, "error": f"RFQ ID '{rfq_id}' not found in database"}), 404
            
        rfq_path_from_db = result[0]
        
        if not rfq_path_from_db:
            return jsonify({"success": False, "error": "RFQ ID found, but 'rfq_file_path' is empty/null"}), 400

        logging.info(f"Found path in DB: {rfq_path_from_db}")

        # --- STEP 2: CONSTRUCT GITHUB URL & DOWNLOAD ---
        # Clean path and encode URL
        clean_path = rfq_path_from_db.strip("/")
        encoded_path = urllib.parse.quote(clean_path)
        
        url = f"https://raw.githubusercontent.com/{GITHUB_OWNER}/{GITHUB_REPO}/{GITHUB_BRANCH}/{encoded_path}"
        
        logging.info(f"Downloading from GitHub: {url}")
        response = requests.get(url, stream=True)

        if response.status_code == 404:
            return jsonify({"success": False, "error": f"File path from DB ({clean_path}) not found on GitHub."}), 404
        elif response.status_code != 200:
            return jsonify({"success": False, "error": f"GitHub Download failed: {response.status_code}"}), 400

        # --- STEP 3: SAVE LOCALLY ---
        original_filename = urllib.parse.unquote(os.path.basename(clean_path))
        safe_filename = secure_filename(original_filename)
        
        # Unique name
        unique_filename = f"{rfq_id}_{int(time.time())}_{safe_filename}"
        local_file_path = UPLOAD_FOLDER / unique_filename

        with open(local_file_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        # --- STEP 4: CONVERT (OCR) ---
        max_pages = int(data.get('max_pages', 20))
        result = convert_PDF(unique_filename, max_pages=max_pages)
        
        # Add metadata to response
        result['rfq_id'] = rfq_id
        result['source_path'] = rfq_path_from_db
        
        return jsonify(result)

    except psycopg2.Error as db_err:
        logging.error(f"Database error: {db_err}")
        return jsonify({"success": False, "error": f"Database error: {str(db_err)}"}), 500
    except Exception as e:
        logging.error(f"Processing error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500
    finally:
        # Close DB connection
        if conn:
            with contextlib.suppress(Exception):
                if cur:
                    cur.close()
                conn.close()
            
        # Cleanup file
        if local_file_path and local_file_path.exists():
            try:
                os.remove(local_file_path)
                logging.info(f"Cleaned up processed file: {local_file_path}")
            except Exception as e:
                logging.warning(f"Failed to cleanup: {e}")


@app.route('/process-rfq-id-to-images', methods=['POST'])
def process_rfq_id_to_images():
    """
    1. Accepts 'rfq_id' in JSON.
    2. Queries Postgres to get 'rfq_file_path'.
    3. Downloads file from Public GitHub.
    4. Converts ALL pages to PNG images.
    5. Returns array of:
        - url  (view image)
        - lien_pour_telecharger_l_image (force-download image)
    """
    if not request.is_json:
        return jsonify({"success": False, "error": "Content-Type must be application/json"}), 400

    data = request.get_json()
    rfq_id = data.get('rfq_id')

    zoom_x = 2.0
    zoom_y = 2.0
    mat = fitz.Matrix(zoom_x, zoom_y)

    if not rfq_id:
        return jsonify({"success": False, "error": "Missing 'rfq_id'"}), 400

    local_pdf_path = None
    conn = None
    cur = None # Added for consistency
    download_url_page_1 = None

    try:
        # --- STEP 1: FETCH PATH FROM DB ---
        logging.info(f"Connecting to DB to fetch path for RFQ ID: {rfq_id}")
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()

        query = "SELECT rfq_file_path FROM public.main WHERE rfq_id = %s"
        cur.execute(query, (rfq_id,))
        result = cur.fetchone()

        if not result or not result[0]:
            return jsonify({
                "success": False,
                "error": f"RFQ ID '{rfq_id}' not found in database or path is empty"
            }), 404

        rfq_path_from_db = result[0]
        logging.info(f"Found path in DB: {rfq_path_from_db}")

        # --- STEP 2: DOWNLOAD FROM GITHUB ---
        clean_path = rfq_path_from_db.strip("/")
        encoded_path = urllib.parse.quote(clean_path)
        url = f"https://raw.githubusercontent.com/{GITHUB_OWNER}/{GITHUB_REPO}/{GITHUB_BRANCH}/{encoded_path}"

        logging.info(f"Downloading from GitHub: {url}")
        response = requests.get(url, stream=True, timeout=30)

        if response.status_code == 404:
            return jsonify({"success": False, "error": f"File not found on GitHub: {clean_path}"}), 404

        elif response.status_code != 200:
            return jsonify({"success": False, "error": f"GitHub download failed, status: {response.status_code}"}), 400

        # --- STEP 3: SAVE PDF LOCALLY ---
        unique_pdf_name = f"{rfq_id}_{int(time.time())}.pdf"
        local_pdf_path = UPLOAD_FOLDER / unique_pdf_name

        with open(local_pdf_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        logging.info(f"PDF saved locally: {local_pdf_path}")

        # --- STEP 4: CONVERT PAGES TO IMAGES ---
        cleanup_old_files(OUTPUT_FOLDER, max_age_hours=24)

        doc = fitz.open(local_pdf_path)
        total_pages = len(doc)

        if total_pages == 0:
            doc.close()
            return jsonify({"success": False, "error": "PDF contains no pages"}), 400

        max_pages_to_convert = int(data.get('max_pages', 20))
        image_urls = []

        timestamp = int(time.time())
        base_url = request.host_url.rstrip('/')

        for i, page in enumerate(doc):
            if i >= max_pages_to_convert:
                break

            pix = page.get_pixmap(matrix=mat)

            image_filename = f"{rfq_id}_page_{i+1}_{timestamp}.png"
            image_save_path = OUTPUT_FOLDER / image_filename

            pix.save(str(image_save_path))

            # --- URLS ---
            view_url = f"{base_url}/images/{image_filename}"
            download_url = f"{base_url}/download-image/{image_filename}"

            image_urls.append({
                "page": i + 1,
                "url": view_url,  # View normally
                "lien_pour_telecharger_l_image": download_url,  # Forces download
                "filename": image_filename
            })

            if i == 0:
                download_url_page_1 = download_url

            logging.info(f"Converted page {i+1}/{min(total_pages, max_pages_to_convert)}")

        doc.close()

        truncated = total_pages > max_pages_to_convert

        # --- STEP 5: RETURN RESULTS ---
        return jsonify({
            "success": True,
            "message": "RFQ PDF converted to images successfully",
            "rfq_id": rfq_id,
            "source_path": rfq_path_from_db,
            "total_pages": total_pages,
            "converted_pages": len(image_urls),
            "truncated": truncated,
            "max_pages": max_pages_to_convert,
            "download_url_page_1_png": download_url_page_1,
            "images": image_urls,
            "note": "Images will be automatically deleted after 24 hours"
        }), 200

    except psycopg2.Error as db_err:
        logging.error(f"Database error: {db_err}")
        return jsonify({"success": False, "error": f"Database error: {str(db_err)}"}), 500

    except Exception as e:
        logging.error(f"Processing error: {e}", exc_info=True)
        return jsonify({"success": False, "error": f"Processing failed: {str(e)}"}), 500

    finally:
        if conn:
            with contextlib.suppress(Exception):
                if cur:
                    cur.close()
                conn.close()

        if local_pdf_path and local_pdf_path.exists():
            try:
                os.remove(local_pdf_path)
                logging.info(f"Cleaned up PDF file: {local_pdf_path}")
            except Exception as e:
                logging.warning(f"Failed to cleanup PDF: {e}")


# --- ENHANCED ENDPOINT: RFQ TO IMAGES + OCR ---

@app.route('/process-rfq-id-to-images-with-ocr', methods=['POST'])
def process_rfq_id_to_images_with_ocr():
    """
    1. Accepts 'rfq_id' in JSON.
    2. Queries Postgres to get 'rfq_file_path'.
    3. Downloads file from Public GitHub.
    4. Converts ALL pages to PNG images.
    5. Runs PaddleOCR on each page.
    6. Returns array with:
        - url (view image)
        - lien_pour_telecharger_l_image (force-download image)
        - ocr_url (annotated image with OCR boxes)
        - ocr_download_url (force-download annotated image)
        - texts (array of detected texts)
        - detailed_ocr (array with text, confidence, box)
    
    Request JSON:
    {
        "rfq_id": "1026205",
        "max_pages": 20,
        "include_ocr_boxes": true,  // Include bounding boxes in response
        "save_annotated": true      // Save annotated images with OCR boxes
    }
    """
    if not request.is_json:
        return jsonify({"success": False, "error": "Content-Type must be application/json"}), 400

    data = request.get_json()
    rfq_id = data.get('rfq_id')

    if not rfq_id:
        return jsonify({"success": False, "error": "Missing 'rfq_id'"}), 400

    # Configuration
    max_pages = int(data.get('max_pages', 20))
    include_ocr_boxes = data.get('include_ocr_boxes', True)
    save_annotated = data.get('save_annotated', True)
    
    zoom_x = 2.0
    zoom_y = 2.0
    mat = fitz.Matrix(zoom_x, zoom_y)

    local_pdf_path = None
    conn = None
    cur = None

    try:
        # --- STEP 1: FETCH PATH FROM DB ---
        logging.info(f"Connecting to DB to fetch path for RFQ ID: {rfq_id}")
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()

        query = "SELECT rfq_file_path FROM public.main WHERE rfq_id = %s"
        cur.execute(query, (rfq_id,))
        result = cur.fetchone()

        if not result or not result[0]:
            return jsonify({
                "success": False,
                "error": f"RFQ ID '{rfq_id}' not found in database or path is empty"
            }), 404

        rfq_path_from_db = result[0]
        logging.info(f"Found path in DB: {rfq_path_from_db}")

        # --- STEP 2: DOWNLOAD FROM GITHUB ---
        clean_path = rfq_path_from_db.strip("/")
        encoded_path = urllib.parse.quote(clean_path)
        url = f"https://raw.githubusercontent.com/{GITHUB_OWNER}/{GITHUB_REPO}/{GITHUB_BRANCH}/{encoded_path}"

        logging.info(f"Downloading from GitHub: {url}")
        response = requests.get(url, stream=True, timeout=30)

        if response.status_code == 404:
            return jsonify({"success": False, "error": f"File not found on GitHub: {clean_path}"}), 404
        elif response.status_code != 200:
            return jsonify({"success": False, "error": f"GitHub download failed, status: {response.status_code}"}), 400

        # --- STEP 3: SAVE PDF LOCALLY ---
        unique_pdf_name = f"{rfq_id}_{int(time.time())}.pdf"
        local_pdf_path = UPLOAD_FOLDER / unique_pdf_name

        with open(local_pdf_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        logging.info(f"PDF saved locally: {local_pdf_path}")

        # --- STEP 4: CONVERT PAGES TO IMAGES & RUN OCR ---
        cleanup_old_files(OUTPUT_FOLDER, max_age_hours=24)

        doc = fitz.open(local_pdf_path)
        total_pages = len(doc)

        if total_pages == 0:
            doc.close()
            return jsonify({"success": False, "error": "PDF contains no pages"}), 400

        pages_data = []
        timestamp = int(time.time())
        base_url = request.host_url.rstrip('/')
        
        total_ocr_time = 0

        for i, page in enumerate(doc):
            if i >= max_pages:
                break

            # Convert page to image
            pix = page.get_pixmap(matrix=mat)
            
            # Save original PNG
            image_filename = f"{rfq_id}_page_{i+1}_{timestamp}.png"
            image_save_path = OUTPUT_FOLDER / image_filename
            pix.save(str(image_save_path))

            # Convert pixmap to numpy array for OCR
            img_data = pix.tobytes("png")
            nparr = np.frombuffer(img_data, np.uint8)
            cv_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            # --- RUN PADDLEOCR ---
            logging.info(f"Running PaddleOCR on page {i+1}/{min(total_pages, max_pages)}")
            ocr_start = time.time()
            
            try:
                ocr_results = run_paddle_ocr_on_image(cv_image)
                ocr_time = round(time.time() - ocr_start, 2)
                total_ocr_time += ocr_time
                
                # Extract texts only
                texts = [entry["text"] for entry in ocr_results]
                
                logging.info(f"Page {i+1}: Found {len(texts)} text elements in {ocr_time}s")
            
            except Exception as ocr_error:
                logging.error(f"OCR failed on page {i+1}: {ocr_error}")
                ocr_results = []
                texts = []
                ocr_time = 0

            # --- SAVE ANNOTATED IMAGE (if requested) ---
            annotated_url = None
            annotated_download_url = None
            
            if save_annotated and ocr_results:
                annotated_filename = f"{rfq_id}_page_{i+1}_{timestamp}_ocr.png"
                save_annotated_paddle_image(cv_image, ocr_results, annotated_filename)
                
                annotated_url = f"{base_url}/images/{annotated_filename}"
                annotated_download_url = f"{base_url}/download-image/{annotated_filename}"

            # --- BUILD PAGE DATA ---
            page_data = {
                "page": i + 1,
                "url": f"{base_url}/images/{image_filename}",
                "lien_pour_telecharger_l_image": f"{base_url}/download-image/{image_filename}",
                "filename": image_filename,
                "texts": texts,
                "total_text_boxes": len(texts),
                "ocr_processing_time_seconds": ocr_time
            }
            
            # Add annotated image URLs if available
            if annotated_url:
                page_data["ocr_url"] = annotated_url
                page_data["ocr_download_url"] = annotated_download_url
                page_data["ocr_filename"] = annotated_filename
            
            # Add detailed OCR results if requested
            if include_ocr_boxes and ocr_results:
                page_data["detailed_ocr"] = ocr_results

            pages_data.append(page_data)

        doc.close()

        truncated = total_pages > max_pages

        # --- STEP 5: RETURN RESULTS ---
        return jsonify({
            "success": True,
            "message": "RFQ PDF converted to images with OCR successfully",
            "rfq_id": rfq_id,
            "source_path": rfq_path_from_db,
            "total_pages": total_pages,
            "converted_pages": len(pages_data),
            "truncated": truncated,
            "max_pages": max_pages,
            "total_ocr_time_seconds": round(total_ocr_time, 2),
            "average_ocr_time_per_page": round(total_ocr_time / len(pages_data), 2) if pages_data else 0,
            "pages": pages_data,
            "note": "Images will be automatically deleted after 24 hours"
        }), 200

    except psycopg2.Error as db_err:
        logging.error(f"Database error: {db_err}")
        return jsonify({"success": False, "error": f"Database error: {str(db_err)}"}), 500

    except Exception as e:
        logging.error(f"Processing error: {e}", exc_info=True)
        return jsonify({"success": False, "error": f"Processing failed: {str(e)}"}), 500

    finally:
        if conn:
            with contextlib.suppress(Exception):
                if cur:
                    cur.close()
                conn.close()

        if local_pdf_path and local_pdf_path.exists():
            try:
                os.remove(local_pdf_path)
                logging.info(f"Cleaned up PDF file: {local_pdf_path}")
            except Exception as e:
                logging.warning(f"Failed to cleanup PDF: {e}")


# --- ALTERNATIVE: LIGHTER VERSION (TEXTS ONLY) ---

@app.route('/process-rfq-id-ocr-only', methods=['POST'])
def process_rfq_id_ocr_only():
    """
    Lighter version: Returns only OCR texts without saving images.
    
    Request JSON:
    {
        "rfq_id": "1026205",
        "max_pages": 20
    }
    
    Response:
    {
        "success": true,
        "rfq_id": "1026205",
        "pages": [
            {
                "page": 1,
                "texts": ["text1", "text2", ...],
                "total_text_boxes": 15
            },
            ...
        ]
    }
    """
    if not request.is_json:
        return jsonify({"success": False, "error": "Content-Type must be application/json"}), 400

    data = request.get_json()
    rfq_id = data.get('rfq_id')

    if not rfq_id:
        return jsonify({"success": False, "error": "Missing 'rfq_id'"}), 400

    max_pages = int(data.get('max_pages', 20))
    
    zoom_x = 2.0
    zoom_y = 2.0
    mat = fitz.Matrix(zoom_x, zoom_y)

    local_pdf_path = None
    conn = None
    cur = None

    try:
        # Fetch from DB
        logging.info(f"Fetching RFQ path for ID: {rfq_id}")
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()

        query = "SELECT rfq_file_path FROM public.main WHERE rfq_id = %s"
        cur.execute(query, (rfq_id,))
        result = cur.fetchone()

        if not result or not result[0]:
            return jsonify({
                "success": False,
                "error": f"RFQ ID '{rfq_id}' not found"
            }), 404

        rfq_path_from_db = result[0]

        # Download from GitHub
        clean_path = rfq_path_from_db.strip("/")
        encoded_path = urllib.parse.quote(clean_path)
        url = f"https://raw.githubusercontent.com/{GITHUB_OWNER}/{GITHUB_REPO}/{GITHUB_BRANCH}/{encoded_path}"

        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()

        # Save PDF
        unique_pdf_name = f"{rfq_id}_{int(time.time())}.pdf"
        local_pdf_path = UPLOAD_FOLDER / unique_pdf_name

        with open(local_pdf_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        # Process pages
        doc = fitz.open(local_pdf_path)
        total_pages = len(doc)

        if total_pages == 0:
            doc.close()
            return jsonify({"success": False, "error": "PDF contains no pages"}), 400

        pages_data = []
        
        for i, page in enumerate(doc):
            if i >= max_pages:
                break

            # Convert page to image in memory
            pix = page.get_pixmap(matrix=mat)

            # Convert pixmap to numpy array for OCR
            img_data = pix.tobytes("png")
            nparr = np.frombuffer(img_data, np.uint8)
            cv_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            # --- RUN PADDLEOCR ---
            ocr_start = time.time()
            try:
                ocr_results = run_paddle_ocr_on_image(cv_image)
                ocr_time = round(time.time() - ocr_start, 2)
                
                texts = [entry["text"] for entry in ocr_results]
                
                logging.info(f"Page {i+1}: Found {len(texts)} text elements in {ocr_time}s")
            
            except Exception as ocr_error:
                logging.error(f"OCR failed on page {i+1}: {ocr_error}")
                texts = []
                ocr_time = 0

            # --- BUILD PAGE DATA ---
            page_data = {
                "page": i + 1,
                "texts": texts,
                "total_text_boxes": len(texts),
                "ocr_processing_time_seconds": ocr_time
            }

            pages_data.append(page_data)

        doc.close()
        truncated = total_pages > max_pages
        
        # --- RETURN RESULTS ---
        return jsonify({
            "success": True,
            "message": "RFQ OCR completed successfully (text only)",
            "rfq_id": rfq_id,
            "source_path": rfq_path_from_db,
            "total_pages": total_pages,
            "processed_pages": len(pages_data),
            "truncated": truncated,
            "max_pages": max_pages,
            "pages": pages_data
        }), 200

    except requests.RequestException as e:
        return jsonify({"success": False, "error": f"GitHub download failed: {str(e)}"}), 400
    except psycopg2.Error as db_err:
        logging.error(f"Database error: {db_err}")
        return jsonify({"success": False, "error": f"Database error: {str(db_err)}"}), 500
    except Exception as e:
        logging.error(f"Processing error: {e}", exc_info=True)
        return jsonify({"success": False, "error": f"Processing failed: {str(e)}"}), 500
    finally:
        if conn:
            with contextlib.suppress(Exception):
                if cur:
                    cur.close()
                conn.close()

        if local_pdf_path and local_pdf_path.exists():
            try:
                os.remove(local_pdf_path)
                logging.info(f"Cleaned up PDF file: {local_pdf_path}")
            except Exception as e:
                logging.warning(f"Failed to cleanup PDF: {e}")


if __name__ == "__main__":
    logging.info("Starting Flask PDF OCR API with Image URL Support")
    use_GPU = torch.cuda.is_available()
    logging.info(f"Using GPU: {use_GPU}")
    logging.info(f"Upload folder: {UPLOAD_FOLDER}")
    logging.info(f"Output folder: {OUTPUT_FOLDER}")
    app.run(host='0.0.0.0', port=5000)
