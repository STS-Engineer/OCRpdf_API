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
    shared-mime-info

# Install requirements (let doctr pull its dependencies including opencv)
echo "INFO: Installing Python requirements..."
pip install --no-cache-dir -r requirements.txt

# Check what OpenCV version was installed and ensure it's working
echo "INFO: Checking OpenCV installation..."
python -c "import cv2; print(f'OpenCV version: {cv2.__version__}')" && echo "✓ OpenCV OK" || {
    echo "✗ OpenCV import failed, trying to fix..."
    # If OpenCV failed, install the latest headless version that's compatible
    pip uninstall -y opencv-python opencv-contrib-python || true
    pip install --force-reinstall opencv-python-headless
    python -c "import cv2; print(f'OpenCV version after reinstall: {cv2.__version__}')" || echo "✗ OpenCV still failed"
}

# Verify other critical installations
echo "INFO: Verifying other installations..."
python -c "import pypdfium2; print('✓ pypdfium2 OK')" || echo "✗ pypdfium2 FAILED"
python -c "import flask; print('✓ Flask OK')" || echo "✗ Flask FAILED"
python -c "import torch; print('✓ PyTorch OK')" || echo "✗ PyTorch FAILED"

# Pre-download NLTK data
echo "INFO: Downloading NLTK stopwords data..."
python -c "import nltk; nltk.download('stopwords', quiet=True)"

# Pre-initialize the OCR model
echo "INFO: Pre-loading OCR model..."
python << 'PYEOF'
import contextlib
import sys
try:
    from doctr.models import ocr_predictor
    print('Loading OCR model...')
    with contextlib.redirect_stdout(None):
        model = ocr_predictor(
            'db_resnet50', 
            'crnn_mobilenet_v3_large', 
            pretrained=True, 
            assume_straight_pages=True
        )
    print('✓ OCR model loaded successfully')
except Exception as e:
    print(f'✗ OCR model loading failed: {e}', file=sys.stderr)
    # Don't exit, let the app start anyway - it will handle model loading
PYEOF

echo "INFO: Starting Gunicorn server..."
exec gunicorn -k gthread -w 2 --threads 4 -b 0.0.0.0:8000 --timeout 300 --preload app:app
