from typing import Generator
import threading
import cv2
from picamera2 import Picamera2
from ultralytics import YOLO
import numpy as np
import time

CONFIDENCE = 0.4  # Confidence threshold for YOLO detections

PERSON_SIZE = 0.184  # Average height of a Playmobil figure in pixel on a 1440p image (1920x1440) at a distance of 50 cm
PERSONE_DISTANCE = (
    50  # Distance in cm at which the Playmobil figure is placed from the camera
)

CAMERA_RESOLUTION = (1920, 1440)  # Camera resolution (width, height)

STREAM_WIDTH = (
    1280  # Stream downscale width (detection still runs on full-res frames)
)
JPEG_QUALITY = 70  # Stream JPEG quality


class VideoCamera:
    def __init__(
        self,
        model: YOLO | str,
        use_model: bool = True,
        imgsz: int = 480,
        inference_interval_ms: float = 500,
        **kwargs,
    ):
        """
        Initializes the VideoCamera object.
        @param model: YOLO model or path to the model weights.
        @param use_model: Whether to use the YOLO model for detection.
        @param inference_interval_ms: Minimum interval in milliseconds between two YOLO inferences.
        @param kwargs: Additional keyword arguments for camera configuration."""

        self.cam = Picamera2()
        cam_config = self.cam.create_preview_configuration(
            main={"format": "RGB888", "size": CAMERA_RESOLUTION},
            queue=kwargs.get("queue", False),
        )
        print(f"Camera sensor resolution: {cam_config["main"]["size"]}")
        print(f"FPS: {self.cam.video_configuration.controls.FrameRate}")
        self.FPS = self.cam.video_configuration.controls.FrameRate
        self.cam.configure(cam_config)
        self.cam.start()

        self.model = (
            YOLO(model, task="detect") if isinstance(model, str) else model
        )  # Load the YOLO model if a path is provided

        # Measurements
        self.playmobile_height_ratio = (
            PERSON_SIZE * cam_config["main"]["size"][1] / 1440
        )

        self.USE_MODEL = use_model  # Set to False to disable YOLO detection
        self.imgsz = imgsz  # Image size for YOLO inference

        self.inference_interval_ms = inference_interval_ms
        self._last_inference_time = 0.0

        # Shared state between the inference thread and the stream consumer
        self._frame_lock = threading.Lock()
        self._latest_frame: np.ndarray | None = None
        self._frame_version = 0
        self._running = True

        self._thread = threading.Thread(
            target=self._inference_loop, daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        """Stops the inference thread."""
        self._running = False
        self._thread.join(timeout=2)

    def _inference_loop(self) -> None:
        """Producer thread: captures frames, runs YOLO inference and keeps the
        latest annotated frame available for the stream consumer."""
        last_result = None

        while self._running:
            frame = self.cam.capture_array()

            run_inference = self.USE_MODEL and (
                time.time() - self._last_inference_time
                >= self.inference_interval_ms / 1000
            )

            if run_inference:
                start_time = time.time()
                results = self.model(
                    frame, imgsz=self.imgsz, classes=[0], verbose=False
                )  # classes=[0] for person detection
                end_time = time.time()
                print(
                    f"Inference time: {(end_time - start_time) * 1000:.4f} ms"
                )
                last_result = results[0]
                self._last_inference_time = time.time()

            if self.USE_MODEL and last_result is not None:
                frame = self._draw_detections(frame, last_result)

            with self._frame_lock:
                self._latest_frame = frame
                self._frame_version += 1

    def _draw_detections(self, frame: np.ndarray, result) -> np.ndarray:
        """Draws the YOLO detections on a copy of the frame."""
        frame = frame.copy()
        for box in result.boxes:
            # Bounding box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Confidence
            conf = float(box.conf[0])

            if conf < CONFIDENCE:  # Filter out low-confidence detections
                continue

            # Class ID and name
            cls = int(box.cls[0])
            box_height_px = y2 - y1
            screen_ratio = box_height_px / frame.shape[0]

            distance = (
                self.playmobile_height_ratio * PERSONE_DISTANCE / screen_ratio
            )  # Calculate distance based on box height
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
        return frame

    def _get_latest_frame(self) -> tuple[np.ndarray, int]:
        """Returns the latest captured frame (and its version), blocking briefly until one is available."""
        while self._running:
            with self._frame_lock:
                if self._latest_frame is not None:
                    return self._latest_frame, self._frame_version
            time.sleep(0.01)
        raise RuntimeError("Camera stopped")

    def get_frame(self) -> bytes:
        """
        Returns the latest frame from the camera as a JPEG-encoded byte string.
        """
        frame, _ = self._get_latest_frame()
        ret, buffer = cv2.imencode(".jpg", frame)
        return buffer.tobytes()

    def generate_frames(self) -> Generator[bytes]:
        """
        Generates frames from the video capture object with YOLO detections drawn on them.

        Runs only when a new frame is produced (never busy-loops), and downscales the
        frame before JPEG encoding so the stream thread stays light on the CPU.
        """
        last_sent_version = -1
        while self._running:
            frame, version = self._get_latest_frame()
            if version == last_sent_version:
                time.sleep(0.05)
                continue
            last_sent_version = version

            if frame.shape[1] > STREAM_WIDTH:
                scale = STREAM_WIDTH / frame.shape[1]
                frame = cv2.resize(
                    frame, (STREAM_WIDTH, round(frame.shape[0] * scale))
                )

            # Encoder en JPEG
            ret, buffer = cv2.imencode(
                ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
            )
            frame_bytes = buffer.tobytes()

            # Envoyer la frame encodée
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
            )
