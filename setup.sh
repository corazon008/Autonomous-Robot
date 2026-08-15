sudo apt install -y ffmpeg libcamera-apps python3-picamera2 python3-pip python-dev-is-python3

rm -rf .venv

uv venv --system-site-packages

uv sync
