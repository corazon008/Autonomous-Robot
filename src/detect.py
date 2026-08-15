from typing import Generator
import cv2
from picamera2 import Picamera2
from ultralytics import YOLO
import numpy as np
import time


#playmobile_height = 7 #cm

# Measurements
playmobile_height_ratio = 0.185 # 0.18
playmobil_distance = 50 #cm

USE_MODEL = False # Set to False to disable YOLO detection

if USE_MODEL:
    model = YOLO("yolo26n_ncnn_model")

def generate_frames(cam: Picamera2)-> Generator[bytes]:
    """
    Generates frames from the video capture object with YOLO detections drawn on them.
    """

    setup = False # bool to print box size

    while True:
        frame = cam.capture_array()

        if USE_MODEL:
            # Inference YOLO
            start_time = time.time()
            results = model(frame, imgsz=32*20, classes=[0], verbose=False) # classes=[0] for person detection
            end_time = time.time()

            print(f"Inference time: {end_time - start_time:.4f} seconds")

            # Dessiner les résultats
            result = results[0]

            for box in result.boxes:
                # Bounding box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Confidence
                conf = float(box.conf[0])

                # Class ID and name
                cls = int(box.cls[0])
                box_height_px = y2 - y1
                screen_ratio = box_height_px / cam.sensor_resolution[1]
                distance = playmobile_height_ratio * playmobil_distance / screen_ratio  # Calculate distance based on box height
                label = f"{model.names[cls]} {conf:.2f} distance : {distance:.2f} cm"

                # Draw rectangle
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Draw label
                cv2.putText(
                    frame,
                    label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )

                if setup:
                    box_height_px = y2 - y1
                    screen_ratio = box_height_px / cam.sensor_resolution[1]
                    print(f"Screen ratio: {screen_ratio:.4f}")

        annotated_frame = frame

        # Encoder en JPEG
        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        frame_bytes = buffer.tobytes()

        # Envoyer la frame encodée
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
