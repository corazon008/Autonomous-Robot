from picamera2 import Picamera2
from picamera2.encoders import H264Encoder
import time

picam2 = Picamera2()

config = picam2.create_video_configuration(
    main={"format": "RGB888", "size": (1280, 960)}
)

picam2.configure(config)

encoder = H264Encoder(bitrate=10_000_000)

picam2.start()
picam2.start_recording(encoder, "video.h264")

time.sleep(30)

picam2.stop_recording()
picam2.stop()
