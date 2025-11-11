#!/usr/bin/env bash
#!/usr/bin/env bash
set -eux

# --- ADD THESE TWO LINES ---
echo "INFO: Updating apt and installing system-level dependencies..."
apt-get update && apt-get install -y libgl1-mesa-glx
# --- END OF ADDED LINES ---

# 1) Install pinned requirements
echo "INFO: Installing Python requirements..."
pip install --no-cache-dir -r requirements.txt

# 2) Ensure GUI OpenCV is removed (docTR may have pulled it back in)
pip uninstall -y opencv-python opencv-contrib-python || true

# 3) Install headless OpenCV + a PDF renderer for DocTR
pip install --no-cache-dir opencv-python-headless==4.10.0.84 pypdfium2==4.30.0

# 5) Run your app
echo "INFO: Starting Gunicorn server..."
exec gunicorn -k gthread -w 2 -b 0.0.0.0:8000 app:app
