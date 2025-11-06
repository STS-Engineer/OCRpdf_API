#!/usr/bin/env bash
set -eux

# 1) Install pinned requirements
pip install --no-cache-dir -r requirements.txt

# 2) Ensure GUI OpenCV is removed (docTR may have pulled it back in)
pip uninstall -y opencv-python opencv-contrib-python || true

# 3) Install headless OpenCV + a PDF renderer for DocTR
pip install --no-cache-dir opencv-python-headless==4.10.0.84 pypdfium2==4.30.0



# 5) Run your app
exec gunicorn -k gthread -w 2 -b 0.0.0.0:8000 app:app
