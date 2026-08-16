from picamera2 import Picamera2
import cv2
import time

WIDTH = 1920
HEIGHT = 1440
FPS = 30
DURATION = 60

picam2 = Picamera2()

config = picam2.create_video_configuration(
    main={
        "format": "BGR888",
        "size": (WIDTH, HEIGHT),
    }
)

picam2.configure(config)
picam2.start()

# Laisse la caméra s'initialiser
time.sleep(1)

fourcc = cv2.VideoWriter_fourcc(*"MJPG")
writer = cv2.VideoWriter(
    "video.avi",
    fourcc,
    FPS,
    (WIDTH, HEIGHT),
)

if not writer.isOpened():
    raise RuntimeError("Impossible d'ouvrir VideoWriter")

print(f"Recording {WIDTH}x{HEIGHT} @ {FPS} FPS for {DURATION}s...")

start = time.monotonic()
frames = 0

while time.monotonic() - start < DURATION:
    frame = picam2.capture_array()

    # Picamera2 RGB888 -> OpenCV BGR
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    writer.write(frame)
    frames += 1

writer.release()
picam2.stop()

elapsed = time.monotonic() - start

print(f"Done.")
print(f"Frames: {frames}")
print(f"Time:   {elapsed:.2f}s")
print(f"Actual FPS: {frames / elapsed:.2f}")
