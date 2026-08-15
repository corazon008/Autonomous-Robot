from typing import Generator
import cv2
from picamera2 import Picamera2
from ultralytics import YOLO
import numpy as np
import time


class VideoCamera:
    def __init__(self, use_model: bool = True, frame_to_skip: int = 30):
        self.cam = Picamera2()
        cam_config = self.cam.create_preview_configuration(main={"format": 'XRGB8888', "size": (640, 480)})
        print(f"Camera sensor resolution: {self.cam.sensor_resolution}")
        self.cam.configure(cam_config)
        self.cam.start()

        self.model = YOLO("yolo26n_ncnn_model")

        # Measurements
        self.playmobile_height_ratio = 0.185 * cam_config["main"]["size"][1] / 1080 # 0.18 for (1920, 1080)
        self.playmobil_distance = 50 #cm

        self.USE_MODEL = use_model # Set to False to disable YOLO detection

        self.FRAME_TO_SKIP = frame_to_skip # Number of frames to skip for YOLO detection (0 means no skipping)
        self.current_frame_nb = 0

    def get_frame(self) -> bytes:
        """
        Returns the current frame from the camera as a JPEG-encoded byte string.
        """
        frame = self.cam.capture_array()
        ret, buffer = cv2.imencode('.jpg', frame)
        return buffer.tobytes()

    def generate_frames(self)-> Generator[bytes]:
        """
        Generates frames from the video capture object with YOLO detections drawn on them.
        """

        setup = False # bool to print box size

        while True:


            frame = self.cam.capture_array()

            if self.USE_MODEL and (self.current_frame_nb % self.FRAME_TO_SKIP == 0):
                # Inference YOLO
                start_time = time.time()
                results = self.model(frame, imgsz=32*20, classes=[0], verbose=False) # classes=[0] for person detection
                end_time = time.time()

                print(f"Inference time: {end_time - start_time:.4f} seconds")

                # Dessiner les résultats
                result = results[0]

            for box in result.boxes:
                # Bounding box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Confidence
                conf = float(box.conf[0])

                if conf < 0.5:  # Filter out low-confidence detections
                    continue

                # Class ID and name
                cls = int(box.cls[0])
                box_height_px = y2 - y1
                screen_ratio = box_height_px / self.cam.sensor_resolution[1]
                distance = self.playmobile_height_ratio * self.playmobil_distance / screen_ratio  # Calculate distance based on box height
                label = f"{self.model.names[cls]} {conf:.2f} distance : {distance:.2f} cm"

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

            self.current_frame_nb += 1
