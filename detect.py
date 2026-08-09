from typing import Generator
import cv2
from picamera2 import Picamera2
from ultralytics import YOLO
import numpy as np
import time


playmobile_height = 7 #cm

# Measurements
playmobile_height_px = 100 #px
playmobil_distance = 50 #cm

model = YOLO("yolo26n.pt")

def generate_frames(cam: Picamera2)-> Generator[bytes]:
    """
    Generates frames from the video capture object with YOLO detections drawn on them.
    """
    while True:
        frame = cam.capture_array()

        # Inference YOLO
        start_time = time.time()
        results = model(frame, imgsz=32*25, classes=[0], verbose=False) # classes=[0] for person detection
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
            label = f"{model.names[cls]} {conf:.2f}"

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

        annotated_frame = frame

        # Encoder en JPEG
        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        frame_bytes = buffer.tobytes()

        # Envoyer la frame encodée
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
