from flask import Flask, jsonify, Response
from pathlib import Path

from servo import Servo
from motor import Ordinary_Car
from videoCamera import VideoCamera

WORKING_DIR = Path(__file__).parent


print("Setting up the camera...")
pwm_servo = Servo()
car = Ordinary_Car()

cam = VideoCamera(model="yolo26n.onnx", use_model=True, frame_to_drop_ratio=0.1)

app = Flask(__name__)


@app.route("/video")
def video():
    return Response(
        cam.generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/")
def index():
    with open(WORKING_DIR / "static" / "index.html", "r") as f:
        return f.read()


@app.route("/command/<command>", methods=["POST"])
def command(command):
    # Camera control commands
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

    # Movement control commands
    elif command == "backward":
        # Handle forward command
        car.left_upper_wheel(1000)
        car.right_upper_wheel(1000)
        car.left_lower_wheel(1000)
        car.right_lower_wheel(1000)
    elif command == "forward":
        # Handle backward command
        car.left_upper_wheel(-1000)
        car.right_upper_wheel(-1000)
        car.left_lower_wheel(-1000)
        car.right_lower_wheel(-1000)
    elif command == "left":
        # Handle left command
        car.left_upper_wheel(-1000)
        car.right_upper_wheel(1000)
        car.left_lower_wheel(-1000)
        car.right_lower_wheel(1000)
    elif command == "right":
        # Handle right command
        car.left_upper_wheel(1000)
        car.right_upper_wheel(-1000)
        car.left_lower_wheel(1000)
        car.right_lower_wheel(-1000)
    elif command == "stop":
        # Handle stop command
        car.left_upper_wheel(0)
        car.right_upper_wheel(0)
        car.left_lower_wheel(0)
        car.right_lower_wheel(0)
    else:
        return jsonify({"status": "error", "message": "Invalid command"}), 400
    return jsonify({"status": "success"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
    print("Server started at http://0.0.0.0:8000")
