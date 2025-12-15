import logging
import time
from pathlib import Path
import contextlib
import os
from datetime import datetime, timedelta
import uuid
import socket
import ipaddress
from urllib.parse import urlparse
import urllib.parse

# ⚠️ CRITICAL: Set these environment variables BEFORE importing PaddleOCR
os.environ['FLAGS_use_mkldnn'] = '0'  # Disable MKL-DNN/OneDNN
os.environ['FLAGS_use_cudnn'] = '0'  # Disable CUDNN (we don't have GPU anyway)

import requests
from werkzeug.utils import secure_filename
from werkzeug.middleware.proxy_fix import ProxyFix
from flask import Flask, request, jsonify, send_file
import nltk
import torch

# --- PADDLEOCR RELATED IMPORTS ---
from paddleocr import PaddleOCR
import cv2
import numpy as np
from typing import List, Dict, Any
# ---------------------------------

import psycopg2
import fitz  # PyMuPDF


# --- CONFIGURATION ---
UPLOAD_FOLDER = Path("/tmp/uploads")
OUTPUT_FOLDER = Path("/tmp/output_images")
ALLOWED_EXTENSIONS = {"pdf", "png", "jpg", "jpeg", "tiff", "bmp"}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

app = Flask(__name__)
app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)

UPLOAD_FOLDER.mkdir(exist_ok=True, parents=True)
OUTPUT_FOLDER.mkdir(exist_ok=True, parents=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["OUTPUT_FOLDER"] = OUTPUT_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024

nltk.download("stopwords", quiet=True)

# --- PADDLEOCR INITIALIZATION ---
paddle_ocr_model = None


def init_paddle_ocr():
    global paddle_ocr_model
    logging.info("Loading PaddleOCR model (lang=ch for Chinese/English/Latin support)...")
    paddle_ocr_model = PaddleOCR(
        lang="ch",
        use_angle_cls=False,
        use_gpu=False,  # Set to False since we're having backend issues
        show_log=False,
        enable_mkldnn=False,  # ⚠️ CRITICAL: Disable OneDNN
        use_dilation=False,   # Additional stability setting
    )
    logging.info("PaddleOCR model loaded successfully (lang='ch').")


try:
    init_paddle_ocr()
except Exception as e:
    logging.error(f"FATAL: Failed to load PaddleOCR during startup: {e}", exc_info=True)

def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def cleanup_old_files(folder: Path, max_age_hours: int = 24):
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


# -------------------------------
# URL -> OpenCV image helpers
# -------------------------------

def _is_public_ip(hostname: str) -> bool:
    """
    Resolve hostname and ensure it doesn't point to private/local/reserved IPs (basic SSRF protection).
    If you need to OCR internal URLs (like localhost), you can disable this check.
    """
    try:
        infos = socket.getaddrinfo(hostname, None)
        for info in infos:
            ip_str = info[4][0]
            ip = ipaddress.ip_address(ip_str)
            if (
                ip.is_private
                or ip.is_loopback
                or ip.is_link_local
                or ip.is_reserved
                or ip.is_multicast
                or ip.is_unspecified
            ):
                return False
        return True
    except Exception:
        return False


def validate_image_url(url: str, allow_localhost: bool = True) -> None:
    p = urlparse(url)
    if p.scheme not in ("http", "https"):
        raise ValueError("Only http/https URLs are allowed")
    if not p.hostname:
        raise ValueError("Invalid URL (missing hostname)")

    # If you are testing with http://localhost:5000/images/xxx.png
    # then allow_localhost=True is needed.
    if allow_localhost and p.hostname in ("localhost", "127.0.0.1"):
        return

    if not _is_public_ip(p.hostname):
        raise ValueError("Blocked URL hostname (non-public/private IP)")


def load_cv_image_from_url(
    url: str,
    timeout: int = 20,
    max_bytes: int = 20 * 1024 * 1024,
    allow_localhost: bool = True,
) -> np.ndarray:
    validate_image_url(url, allow_localhost=allow_localhost)

    r = requests.get(url, stream=True, timeout=timeout, allow_redirects=True)
    r.raise_for_status()

    data = bytearray()
    for chunk in r.iter_content(chunk_size=1024 * 64):
        if not chunk:
            continue
        data.extend(chunk)
        if len(data) > max_bytes:
            raise ValueError(f"Image too large (> {max_bytes} bytes)")

    nparr = np.frombuffer(bytes(data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Failed to decode image from URL")
    return img


# -------------------------------
# PaddleOCR helpers
# -------------------------------

def prepare_image_for_paddle(img: np.ndarray, target_max_side: int = 2200) -> tuple:
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


def preprocess_image_for_ocr(image: np.ndarray) -> np.ndarray:
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    processed = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )

    processed = cv2.fastNlMeansDenoising(processed, None, 10, 7, 21)
    processed = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
    return processed


def run_paddle_ocr_on_image(image: np.ndarray, preprocess: bool = True) -> List[Dict[str, Any]]:
    global paddle_ocr_model
    if paddle_ocr_model is None:
        raise RuntimeError("PaddleOCR model not initialized")

    if preprocess:
        logging.info("Preprocessing image for better OCR...")
        image = preprocess_image_for_ocr(image)

    processed_img, scale = prepare_image_for_paddle(image)

    # ✅ Prefer numpy input
    try:
        logging.info("Running PaddleOCR (numpy array input)...")
        result = paddle_ocr_model.ocr(processed_img, cls=False)
    except Exception as e:
        logging.warning(f"Numpy input failed -> fallback to temp file. Reason: {e}")
        temp_path = UPLOAD_FOLDER / f"temp_paddle_{int(time.time())}_{uuid.uuid4().hex[:8]}.png"
        cv2.imwrite(str(temp_path), processed_img)
        try:
            result = paddle_ocr_model.ocr(str(temp_path), cls=False)
        finally:
            with contextlib.suppress(Exception):
                if temp_path.exists():
                    os.remove(temp_path)

    entries: List[Dict[str, Any]] = []

    if not result or not result[0]:
        return entries

    for idx, line in enumerate(result[0]):
        if not line:
            continue
        try:
            box = line[0]
            text_info = line[1]

            text = text_info[0]
            confidence = float(text_info[1])

            if not text or not text.strip():
                continue

            box_pts = normalize_poly(box, scale=scale)
            if len(box_pts) < 3:
                continue

            entries.append({"text": str(text), "confidence": confidence, "box": box_pts})
        except Exception as line_error:
            logging.error(f"Error processing line {idx}: {line_error}")
            continue

    return entries


def save_annotated_paddle_image(image: np.ndarray, entries: List[Dict[str, Any]], output_filename: str) -> str:
    annotated = image.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX

    for idx, entry in enumerate(entries, start=1):
        box = entry.get("box")
        text = entry.get("text", "")

        if not box or len(box) < 3:
            continue

        pts = np.array(box, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(annotated, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

        x, y = box[0]
        cv2.putText(annotated, f"{idx}", (x, y - 10), font, 0.5, (0, 0, 255), 2, lineType=cv2.LINE_AA)

        display_text = text[:20] + "..." if len(text) > 20 else text
        cv2.putText(annotated, display_text, (x, y + 20), font, 0.35, (255, 0, 0), 1, lineType=cv2.LINE_AA)

    output_path = OUTPUT_FOLDER / output_filename
    cv2.imwrite(str(output_path), annotated)
    logging.info(f"Saved annotated image to: {output_path}")
    return output_filename


# --- INFO & HEALTH ENDPOINTS ---
@app.route("/")
def index():
    return jsonify({
        "message": "PDF OCR API with PaddleOCR",
        "version": "6.0 (PaddleOCR from URL supported)",
        "endpoints": {
            "/process-rfq-id-to-images-with-ocr": "POST",
            "/process-rfq-id-ocr-only": "POST",
            "/test-ocr-on-image": "POST (image_url or image_filename)",
            "/ocr-from-image-url": "POST (image_url)",
            "/images/<filename>": "GET",
            "/download-image/<filename>": "GET",
            "/cleanup": "POST",
            "/health": "GET",
        }
    })


@app.route("/health")
def health():
    return jsonify({
        "status": "healthy",
        "paddle_ocr_model_loaded": paddle_ocr_model is not None,
        "gpu_available": torch.cuda.is_available(),
        "upload_folder": str(UPLOAD_FOLDER),
        "output_folder": str(OUTPUT_FOLDER),
    })


# --- NEW: OCR DIRECTLY FROM IMAGE URL ---
@app.route("/ocr-from-image-url", methods=["POST"])
def ocr_from_image_url():
    if not request.is_json:
        return jsonify({"success": False, "error": "Content-Type must be application/json"}), 400

    data = request.get_json()
    image_url = data.get("image_url")
    preprocess = data.get("preprocess", True)

    if not image_url:
        return jsonify({"success": False, "error": "Missing 'image_url'"}), 400

    try:
        cv_image = load_cv_image_from_url(image_url, allow_localhost=True)
        ocr_results = run_paddle_ocr_on_image(cv_image, preprocess=preprocess)
        texts = [e["text"] for e in ocr_results]

        return jsonify({
            "success": True,
            "image_url": image_url,
            "texts": texts,
            "total_text_boxes": len(texts),
            "detailed_ocr": ocr_results
        }), 200

    except Exception as e:
        logging.error(f"OCR-from-URL error: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500

# --- UPDATED: TEST OCR ON IMAGE (URL OR LOCAL FILE) ---
@app.route("/test-ocr-on-image", methods=["POST"])
def test_ocr_on_image():
    """
    Request JSON (URL):
    {
      "image_url": "http://localhost:5000/images/xxx.png",
      "preprocess": true,
      "save_debug_image": true
    }

    Or (local file in OUTPUT_FOLDER):
    {
      "image_filename": "xxx.png",
      "preprocess": true,
      "save_debug_image": true
    }
    """
    if not request.is_json:
        return jsonify({"success": False, "error": "Content-Type must be application/json"}), 400

    data = request.get_json()
    image_url = data.get("image_url")
    image_filename = data.get("image_filename")
    preprocess = data.get("preprocess", True)
    save_debug = data.get("save_debug_image", True)

    if not image_url and not image_filename:
        return jsonify({"success": False, "error": "Missing 'image_url' or 'image_filename'"}), 400

    try:
        if image_url:
            cv_image = load_cv_image_from_url(image_url, allow_localhost=True)
            source = {"type": "url", "value": image_url}
        else:
            safe_filename = secure_filename(image_filename)
            image_path = OUTPUT_FOLDER / safe_filename
            if not image_path.exists():
                return jsonify({"success": False, "error": f"Image not found: {image_filename}"}), 404
            cv_image = cv2.imread(str(image_path))
            if cv_image is None:
                return jsonify({"success": False, "error": "Failed to load image"}), 400
            source = {"type": "file", "value": safe_filename}

        debug_preprocessed_url = None
        if save_debug and preprocess:
            base_url = request.host_url.rstrip("/")
            debug_img = preprocess_image_for_ocr(cv_image)
            debug_filename = f"debug_preprocessed_{int(time.time())}_{uuid.uuid4().hex[:6]}.png"
            cv2.imwrite(str(OUTPUT_FOLDER / debug_filename), debug_img)
            debug_preprocessed_url = f"{base_url}/images/{debug_filename}"

        ocr_start = time.time()
        ocr_results = run_paddle_ocr_on_image(cv_image, preprocess=preprocess)
        ocr_time = round(time.time() - ocr_start, 2)
        texts = [entry["text"] for entry in ocr_results]

        annotated_url = None
        if ocr_results:
            base_url = request.host_url.rstrip("/")
            annotated_filename = f"debug_annotated_{int(time.time())}_{uuid.uuid4().hex[:6]}.png"
            save_annotated_paddle_image(cv_image, ocr_results, annotated_filename)
            annotated_url = f"{base_url}/images/{annotated_filename}"

        return jsonify({
            "success": True,
            "source": source,
            "image_shape": cv_image.shape,
            "preprocessing_enabled": preprocess,
            "texts": texts,
            "total_text_boxes": len(texts),
            "detailed_ocr": ocr_results,
            "ocr_processing_time_seconds": ocr_time,
            "debug_preprocessed_image_url": debug_preprocessed_url,
            "annotated_image_url": annotated_url,
        }), 200

    except Exception as e:
        logging.error(f"Test OCR error: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500


# --- IMAGE SERVING ENDPOINTS ---
@app.route("/images/<filename>", methods=["GET"])
def serve_image(filename):
    try:
        safe_filename = secure_filename(filename)
        image_path = OUTPUT_FOLDER / safe_filename

        if not image_path.exists():
            return jsonify({"success": False, "error": "Image not found"}), 404

        mime_type = "image/png" if safe_filename.lower().endswith(".png") else "image/jpeg"
        return send_file(str(image_path), mimetype=mime_type, as_attachment=False, download_name=safe_filename)
    except Exception as e:
        logging.error(f"Error serving image: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/download-image/<filename>", methods=["GET"])
def download_image(filename):
    try:
        safe_filename = secure_filename(filename)
        image_path = OUTPUT_FOLDER / safe_filename

        if not image_path.exists():
            return jsonify({"success": False, "error": "Image not found"}), 404

        if safe_filename.lower().endswith(".png"):
            mime_type = "image/png"
        elif safe_filename.lower().endswith((".jpg", ".jpeg")):
            mime_type = "image/jpeg"
        else:
            mime_type = "application/octet-stream"

        return send_file(str(image_path), mimetype=mime_type, as_attachment=True, download_name=safe_filename)
    except Exception as e:
        logging.error(f"Error downloading image: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


# --- CLEANUP ENDPOINT ---
@app.route("/cleanup", methods=["POST"])
def manual_cleanup():
    try:
        max_age = int(request.get_json(silent=True).get("max_age_hours", 1)) if request.is_json else 1
        cleanup_old_files(OUTPUT_FOLDER, max_age_hours=max_age)
        cleanup_old_files(UPLOAD_FOLDER, max_age_hours=max_age)
        return jsonify({"success": True, "message": f"Cleanup completed (files older than {max_age} hours removed)"}), 200
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# --- EXTERNAL CONFIG (use env vars, do NOT hardcode secrets) ---
GITHUB_OWNER = os.getenv("GITHUB_OWNER", "STS-Engineer")
GITHUB_REPO = os.getenv("GITHUB_REPO", "RFQ-back")
GITHUB_BRANCH = os.getenv("GITHUB_BRANCH", "main")
CERT_PATH = "/path/to/your/downloaded/DigiCertGlobalRootG2.crt.pem"
DB_CONFIG = {
    "host": "avo-adb-002.postgres.database.azure.com",
    "database": "RFQ_DATA",
    # Make sure the environment variables are set to the correct values
    "user": os.getenv("RFQ_DB_USER", "administrationSTS"),
    "password": os.getenv("RFQ_DB_PASSWORD", "St$@0987"),
    # RECOMMENDED CHANGE: Use verify-full and provide the certificate path
    "sslmode": os.getenv("RFQ_DB_SSLMODE", "verify-full"),
    "sslrootcert": CERT_PATH  # NEW REQUIRED PARAMETER
}

# --- MAIN ENDPOINT: RFQ TO IMAGES + OCR ---
@app.route("/process-rfq-id-to-images-with-ocr", methods=["POST"])
def process_rfq_id_to_images_with_ocr():
    if not request.is_json:
        return jsonify({"success": False, "error": "Content-Type must be application/json"}), 400

    data = request.get_json()
    rfq_id = data.get("rfq_id")
    if not rfq_id:
        return jsonify({"success": False, "error": "Missing 'rfq_id'"}), 400

    max_pages = int(data.get("max_pages", 20))
    include_ocr_boxes = data.get("include_ocr_boxes", True)
    save_annotated = data.get("save_annotated", True)
    preprocess = data.get("preprocess_images", True)
    zoom_factor = float(data.get("zoom_factor", 3.0))
    mat = fitz.Matrix(zoom_factor, zoom_factor)

    local_pdf_path = None
    conn = None
    cur = None

    try:
        # DB fetch path
        logging.info(f"Connecting to DB to fetch path for RFQ ID: {rfq_id}")
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()
        cur.execute("SELECT rfq_file_path FROM public.main WHERE rfq_id = %s", (rfq_id,))
        result = cur.fetchone()

        if not result or not result[0]:
            return jsonify({"success": False, "error": f"RFQ ID '{rfq_id}' not found in database or path is empty"}), 404

        rfq_path_from_db = result[0]
        logging.info(f"Found path in DB: {rfq_path_from_db}")

        # download from github
        clean_path = rfq_path_from_db.strip("/")
        encoded_path = urllib.parse.quote(clean_path)
        url = f"https://raw.githubusercontent.com/{GITHUB_OWNER}/{GITHUB_REPO}/{GITHUB_BRANCH}/{encoded_path}"

        logging.info(f"Downloading from GitHub: {url}")
        response = requests.get(url, stream=True, timeout=30)
        if response.status_code == 404:
            return jsonify({"success": False, "error": f"File not found on GitHub: {clean_path}"}), 404
        if response.status_code != 200:
            return jsonify({"success": False, "error": f"GitHub download failed, status: {response.status_code}"}), 400

        unique_pdf_name = f"{rfq_id}_{int(time.time())}.pdf"
        local_pdf_path = UPLOAD_FOLDER / unique_pdf_name
        with open(local_pdf_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        doc = fitz.open(local_pdf_path)
        total_pages = len(doc)
        if total_pages == 0:
            doc.close()
            return jsonify({"success": False, "error": "PDF contains no pages"}), 400

        cleanup_old_files(OUTPUT_FOLDER, max_age_hours=24)

        pages_data = []
        timestamp = int(time.time())
        base_url = request.host_url.rstrip("/")
        total_ocr_time = 0.0

        for i, page in enumerate(doc):
            if i >= max_pages:
                break

            pix = page.get_pixmap(matrix=mat)

            image_filename = f"{rfq_id}_page_{i+1}_{timestamp}.png"
            image_save_path = OUTPUT_FOLDER / image_filename
            pix.save(str(image_save_path))

            # in-memory numpy
            img_data = pix.tobytes("png")
            nparr = np.frombuffer(img_data, np.uint8)
            cv_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            logging.info(f"Running PaddleOCR on page {i+1}")
            ocr_start = time.time()
            try:
                ocr_results = run_paddle_ocr_on_image(cv_image, preprocess=preprocess)
                ocr_time = round(time.time() - ocr_start, 2)
                total_ocr_time += ocr_time
                texts = [e["text"] for e in ocr_results]
            except Exception as ocr_error:
                logging.error(f"OCR failed on page {i+1}: {ocr_error}", exc_info=True)
                ocr_results = []
                texts = []
                ocr_time = 0

            annotated_url = None
            annotated_download_url = None
            annotated_filename = None

            if save_annotated and ocr_results:
                annotated_filename = f"{rfq_id}_page_{i+1}_{timestamp}_ocr.png"
                save_annotated_paddle_image(cv_image, ocr_results, annotated_filename)
                annotated_url = f"{base_url}/images/{annotated_filename}"
                annotated_download_url = f"{base_url}/download-image/{annotated_filename}"

            page_data = {
                "page": i + 1,
                "url": f"{base_url}/images/{image_filename}",
                "lien_pour_telecharger_l_image": f"{base_url}/download-image/{image_filename}",
                "filename": image_filename,
                "texts": texts,
                "total_text_boxes": len(texts),
                "ocr_processing_time_seconds": ocr_time,
            }

            if annotated_url:
                page_data["ocr_url"] = annotated_url
                page_data["ocr_download_url"] = annotated_download_url
                page_data["ocr_filename"] = annotated_filename

            if include_ocr_boxes and ocr_results:
                page_data["detailed_ocr"] = ocr_results

            pages_data.append(page_data)

        doc.close()
        truncated = total_pages > max_pages

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
            "note": "Images will be automatically deleted after 24 hours",
        }), 200

    except psycopg2.Error as db_err:
        logging.error(f"Database error: {db_err}", exc_info=True)
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
            with contextlib.suppress(Exception):
                os.remove(local_pdf_path)


# --- LIGHTWEIGHT VERSION: TEXT ONLY ---
@app.route("/process-rfq-id-ocr-only", methods=["POST"])
def process_rfq_id_ocr_only():
    if not request.is_json:
        return jsonify({"success": False, "error": "Content-Type must be application/json"}), 400

    data = request.get_json()
    rfq_id = data.get("rfq_id")
    if not rfq_id:
        return jsonify({"success": False, "error": "Missing 'rfq_id'"}), 400

    max_pages = int(data.get("max_pages", 20))
    zoom_x = 2.0
    zoom_y = 2.0
    mat = fitz.Matrix(zoom_x, zoom_y)

    local_pdf_path = None
    conn = None
    cur = None

    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()
        cur.execute("SELECT rfq_file_path FROM public.main WHERE rfq_id = %s", (rfq_id,))
        result = cur.fetchone()

        if not result or not result[0]:
            return jsonify({"success": False, "error": f"RFQ ID '{rfq_id}' not found"}), 404

        rfq_path_from_db = result[0]

        clean_path = rfq_path_from_db.strip("/")
        encoded_path = urllib.parse.quote(clean_path)
        url = f"https://raw.githubusercontent.com/{GITHUB_OWNER}/{GITHUB_REPO}/{GITHUB_BRANCH}/{encoded_path}"

        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()

        unique_pdf_name = f"{rfq_id}_{int(time.time())}.pdf"
        local_pdf_path = UPLOAD_FOLDER / unique_pdf_name
        with open(local_pdf_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        doc = fitz.open(local_pdf_path)
        total_pages = len(doc)
        if total_pages == 0:
            doc.close()
            return jsonify({"success": False, "error": "PDF contains no pages"}), 400

        pages_data = []
        for i, page in enumerate(doc):
            if i >= max_pages:
                break

            pix = page.get_pixmap(matrix=mat)
            img_data = pix.tobytes("png")
            nparr = np.frombuffer(img_data, np.uint8)
            cv_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            ocr_start = time.time()
            try:
                ocr_results = run_paddle_ocr_on_image(cv_image, preprocess=True)
                ocr_time = round(time.time() - ocr_start, 2)
                texts = [entry["text"] for entry in ocr_results]
            except Exception as ocr_error:
                logging.error(f"OCR failed on page {i+1}: {ocr_error}", exc_info=True)
                texts = []
                ocr_time = 0

            pages_data.append({
                "page": i + 1,
                "texts": texts,
                "total_text_boxes": len(texts),
                "ocr_processing_time_seconds": ocr_time,
            })

        doc.close()
        truncated = total_pages > max_pages

        return jsonify({
            "success": True,
            "message": "RFQ OCR completed successfully (text only)",
            "rfq_id": rfq_id,
            "source_path": rfq_path_from_db,
            "total_pages": total_pages,
            "processed_pages": len(pages_data),
            "truncated": truncated,
            "max_pages": max_pages,
            "pages": pages_data,
        }), 200

    except requests.RequestException as e:
        return jsonify({"success": False, "error": f"GitHub download failed: {str(e)}"}), 400
    except psycopg2.Error as db_err:
        logging.error(f"Database error: {db_err}", exc_info=True)
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
            with contextlib.suppress(Exception):
                os.remove(local_pdf_path)


if __name__ == "__main__":
    logging.info("Starting Flask PDF OCR API with PaddleOCR")
    logging.info(f"Using GPU: {torch.cuda.is_available()}")
    logging.info(f"Upload folder: {UPLOAD_FOLDER}")
    logging.info(f"Output folder: {OUTPUT_FOLDER}")
    app.run(host="0.0.0.0", port=5000, debug=True)
