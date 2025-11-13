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
    python3-opencv  # Add system OpenCV package as backup

# Install pinned requirements
echo "INFO: Installing Python requirements..."
pip install --no-cache-dir -r requirements.txt

# Reinstall OpenCV headless explicitly with dependencies
echo "INFO: Reinstalling OpenCV headless..."
pip uninstall -y opencv-python opencv-contrib-python || true
pip install --no-cache-dir --force-reinstall opencv-python-headless==4.10.0.84

# Install pypdfium2 separately
echo "INFO: Installing pypdfium2..."
pip install --no-cache-dir pypdfium2==4.30.0

# Verify installation
echo "INFO: Verifying OpenCV installation..."
python -c "import cv2; print(f'OpenCV version: {cv2.__version__}')"

# Pre-download NLTK data
echo "INFO: Downloading NLTK stopwords data..."
python -c "import nltk; nltk.download('stopwords', quiet=True)"

# Pre-initialize the OCR model to avoid cold start issues
echo "INFO: Pre-loading OCR model..."
python -c "
import contextlib
from doctr.models import ocr_predictor
print('Loading OCR model...')
with contextlib.redirect_stdout(None):
    model = ocr_predictor('db_resnet50', 'crnn_mobilenet_v3_large', pretrained=True, assume_straight_pages=True)
print('OCR model loaded successfully')
" || echo "WARNING: Could not pre-load OCR model"

# Run your app with increased timeout for model loading
echo "INFO: Starting Gunicorn server..."
exec gunicorn -k gthread -w 2 --threads 4 -b 0.0.0.0:8000 --timeout 300 --preload app:app
