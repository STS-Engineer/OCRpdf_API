#!/usr/bin/env bash
set -eux

# --- ADD THESE TWO LINES ---
echo "INFO: Updating apt and installing system-level dependencies..."
apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0
# --- END OF ADDED LINES ---

# 1) Install pinned requirements
echo "INFO: Installing Python requirements..."
pip install --no-cache-dir -r requirements.txt

# 2) Ensure GUI OpenCV is removed (docTR may have pulled it back in)
echo "INFO: Removing any non-headless OpenCV installations..."
pip uninstall -y opencv-python opencv-contrib-python || true

# 3) Install headless OpenCV + a PDF renderer for DocTR
echo "INFO: Installing opencv-python-headless and pypdfium2..."
pip install --no-cache-dir opencv-python-headless==4.10.0.84 pypdfium2==4.30.0

# 4) Verify installation
echo "INFO: Verifying OpenCV installation..."
python -c "import cv2; print(f'OpenCV version: {cv2.__version__}')" || echo "WARNING: OpenCV import failed"

# 5) Run your app
echo "INFO: Starting Gunicorn server..."
exec gunicorn -k gthread -w 2 -b 0.0.0.0:8000 app:app

