# Add these imports at the top of your Flask app file (if not already added)
from paddleocr import PaddleOCR
import cv2
import numpy as np
from typing import List, Dict, Any, Optional

# --- PADDLEOCR INITIALIZATION ---
paddle_ocr_model = None

def init_paddle_ocr():
    """Initialize PaddleOCR model"""
    global paddle_ocr_model
    logging.info("Loading PaddleOCR model...")
    try:
        paddle_ocr_model = PaddleOCR(
            lang="ch",  # Change to "en" for English or other languages
            use_angle_cls=False,
            use_gpu=torch.cuda.is_available()
        )
        logging.info("PaddleOCR model loaded successfully")
    except Exception as e:
        logging.error(f"Failed to load PaddleOCR model: {e}", exc_info=True)

# Pre-load PaddleOCR at module level
try:
    init_paddle_ocr()
except Exception as e:
    logging.error(f"FATAL: Failed to load PaddleOCR during startup: {e}", exc_info=True)


# --- HELPER FUNCTIONS ---

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
        confidence = entry.get("confidence", 0)
        
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
        "save_annotated": true       // Save annotated images with OCR boxes
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

            # Convert to image
            pix = page.get_pixmap(matrix=mat)
            img_data = pix.tobytes("png")
            nparr = np.frombuffer(img_data, np.uint8)
            cv_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            # Run OCR
            try:
                ocr_results = run_paddle_ocr_on_image(cv_image)
                texts = [entry["text"] for entry in ocr_results]
            except Exception as ocr_error:
                logging.error(f"OCR failed on page {i+1}: {ocr_error}")
                texts = []

            pages_data.append({
                "page": i + 1,
                "texts": texts,
                "total_text_boxes": len(texts)
            })

        doc.close()

        return jsonify({
            "success": True,
            "rfq_id": rfq_id,
            "source_path": rfq_path_from_db,
            "total_pages": total_pages,
            "processed_pages": len(pages_data),
            "truncated": total_pages > max_pages,
            "pages": pages_data
        }), 200

    except Exception as e:
        logging.error(f"Processing error: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500

    finally:
        if conn:
            with contextlib.suppress(Exception):
                cur.close()
                conn.close()

        if local_pdf_path and local_pdf_path.exists():
            try:
                os.remove(local_pdf_path)
            except Exception as e:
                logging.warning(f"Failed to cleanup PDF: {e}")
