#!/usr/bin/env bash
set -eux

# Install system-level dependencies
echo "INFO: Updating apt and installing system-level dependencies..."
apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libpango-1.0-0 \
    libpangocairo-1.0-0 \
    libgdk-pixbuf2.0-0 \
    libffi-dev \
    shared-mime-info \
    poppler-utils \
    poppler-data \
    tesseract-ocr # Ensure Tesseract is available if other parts of code rely on it

# Optional: verify poppler installed
echo "INFO: Verifying poppler installation..."
if command -v pdfinfo >/dev/null 2>&1; then
    pdfinfo -v || true
    echo "✓ Poppler (pdfinfo) available"
else
    echo "⚠ WARNING: pdfinfo not found in PATH"
fi

# 1. Clean up any existing OpenCV installations to start fresh
echo "INFO: Cleaning up any existing OpenCV installations..."
pip uninstall -y opencv-python opencv-contrib-python opencv-python-headless || true

# 2. Install Python requirements (Ensure 'paddleocr' and 'paddlepaddle' are in requirements.txt)
echo "INFO: Installing Python requirements (including docTR and PaddleOCR dependencies)..."
pip install --no-cache-dir -r requirements.txt
# This step installs all dependencies, including docTR (which may pull a non-headless OpenCV) 
# and PaddleOCR (which requires OpenCV).

# 3. Force remove any GUI opencv that might have been installed by docTR/Paddle dependencies
echo "INFO: Removing non-headless OpenCV versions..."
pip uninstall -y opencv-python opencv-contrib-python || true

# 4. Install the required headless version (LAST STEP) to ensure it is the active 'cv2'
echo "INFO: Installing the pinned opencv-python-headless version..."
# Pinning to a recent stable version is generally recommended
pip install --no-cache-dir opencv-python-headless==4.10.0.84

# 5. Install pypdfium2 (LAST STEP)
echo "INFO: Installing pypdfium2..."
pip install --no-cache-dir pypdfium2

# Verify installation
echo "INFO: Verifying installations..."
python -c "import sys; print(f'Python: {sys.version}')"
python -c "import cv2; print(f'OpenCV: {cv2.__version__}')" && echo "✓ OpenCV OK" || echo "✗ OpenCV FAILED"
python -c "import pypdfium2; print('✓ pypdfium2 OK')" || echo "✗ pypdfium2 FAILED"
python -c "import flask; print('✓ Flask OK')" || echo "✗ Flask FAILED"
python -c "import torch; print('✓ PyTorch OK')" || echo "✗ PyTorch FAILED"
python -c "import paddleocr; print('✓ PaddleOCR OK')" || echo "✗ PaddleOCR FAILED"

# Pre-download NLTK data
echo "INFO: Downloading NLTK stopwords data..."
python -c "import nltk; nltk.download('stopwords', quiet=True)"

# 7. Pre-initialize the docTR OCR model
echo "INFO: Pre-loading docTR OCR model..."
python << 'PYEOF_DOCTR'
import contextlib
import sys
try:
    from doctr.models import ocr_predictor
    print('Loading docTR OCR model...')
    with contextlib.redirect_stdout(None):
        model = ocr_predictor(
            'db_resnet50',
            'crnn_mobilenet_v3_large',
            pretrained=True,
            assume_straight_pages=True
        )
    print('✓ docTR OCR model loaded successfully')
except Exception as e:
    print(f'✗ docTR OCR model loading failed: {e}', file=sys.stderr)
PYEOF_DOCTR

# 8. Pre-initialize the PaddleOCR model
echo "INFO: Pre-loading PaddleOCR model..."
python << 'PYEOF_PADDLE'
import sys
import contextlib
try:
    from paddleocr import PaddleOCR
    import torch
    print('Loading PaddleOCR model (lang=ch)...')
    # This must match the initialization logic in app.py
    with contextlib.redirect_stdout(None):
        model = PaddleOCR(
            lang="ch", 
            use_angle_cls=False,
            use_gpu=torch.cuda.is_available()
        )
    print('✓ PaddleOCR model loaded successfully')
except Exception as e:
    print(f'✗ PaddleOCR model loading failed: {e}', file=sys.stderr)
PYEOF_PADDLE

echo "INFO: Starting Gunicorn server..."
exec gunicorn -k gthread -w 2 --threads 4 -b 0.0.0.0:8000 --timeout 300 --preload app:app
