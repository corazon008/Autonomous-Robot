import cv2

def generate_frames(cap):
    print("Generator created")

    while True:
        print("Loop")

        frame = cap.capture_array()

        print("Process")

        annotated_frame = frame

        # Encoder en JPEG
        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        frame_bytes = buffer.tobytes()

        # Envoyer la frame encodée
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
