import cv2
from flask import Flask, jsonify, Response
from picamera2 import Picamera2
from libcamera import Transform
import detect
from pathlib import Path

from servo import Servo

WORKING_DIR = Path(__file__).parent

print("Setting up the camera...")
pwm_servo = Servo()

picam2 = Picamera2()
picam2.configure(
    picam2.create_preview_configuration(
        main={"size": (1920, 1080), "format": "RGB888"},
        #transform=Transform(hflip=1, vflip=1)
    )
)
picam2.start()

app = Flask(__name__)

@app.route('/video')
def video():
    return Response(detect.generate_frames(picam2),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    with open(WORKING_DIR / 'static' / 'index.html', 'r') as f:
        return f.read()

@app.route('/command/<command>', methods=['POST'])
def command(command):
    if command == "camera_left":
        # Handle camera left command
        pwm_servo.move_left_incremental(5)
    elif command == "camera_right":
        # Handle camera right command
        pwm_servo.move_right_incremental(5)
    elif command == "camera_up":
        # Handle camera up command
        pwm_servo.move_up_incremental(5)
    elif command == "camera_down":
        # Handle camera down command
        pwm_servo.move_down_incremental(5)
    # Handle the command logic here
    return jsonify({"status": "success"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
    print("Server started at http://0.0.0.0:8000")
