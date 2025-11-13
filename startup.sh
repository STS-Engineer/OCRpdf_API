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

# Remove any existing OpenCV installations
echo "INFO: Cleaning up any existing OpenCV installations..."
pip uninstall -y opencv-python opencv-contrib-python opencv-python-headless || true

# Install opencv-headless FIRST
echo "INFO: Installing opencv-python-headless..."
pip install --no-cache-dir opencv-python-headless==4.10.0.84

# Install pypdfium2
echo "INFO: Installing pypdfium2..."
pip install --no-cache-dir pypdfium2==4.30.0

# Now install requirements without opencv
echo "INFO: Installing Python requirements (without opencv)..."
pip install --no-cache-dir -r requirements.txt || true

# Force remove any GUI opencv that might have been installed as dependency
echo "INFO: Ensuring only headless OpenCV is present..."
pip uninstall -y opencv-python opencv-contrib-python || true

# Reinstall headless if it was removed
pip list | grep opencv || pip install --no-cache-dir opencv-python-headless==4.10.0.84

# Verify installation
echo "INFO: Verifying installations..."
python -c "import sys; print(f'Python: {sys.version}')"
python -c "import cv2; print(f'OpenCV: {cv2.__version__}')" && echo "✓ OpenCV OK" || echo "✗ OpenCV FAILED"
python -c "import pypdfium2; print('✓ pypdfium2 OK')" || echo "✗ pypdfium2 FAILED"
python -c "import flask; print('✓ Flask OK')" || echo "✗ Flask FAILED"
python -c "import torch; print('✓ PyTorch OK')" || echo "✗ PyTorch FAILED"

# Pre-download NLTK data
echo "INFO: Downloading NLTK stopwords data..."
python -c "import nltk; nltk.download('stopwords', quiet=True)" || true

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
PYEOF

echo "INFO: Starting Gunicorn server..."
exec gunicorn -k gthread -w 2 --threads 4 -b 0.0.0.0:8000 --timeout 300 --preload app:app
