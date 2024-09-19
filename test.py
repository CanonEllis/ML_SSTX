from flask import Flask, Response
import cv2

app = Flask(__name__)

# Capture video from the Raspberry Pi camera (or USB camera)
camera = cv2.VideoCapture(0)  # Use 0 for the default camera

def generate_frames():
    while True:
        # Read a frame from the camera
        success, frame = camera.read()
        if not success:
            break
        else:
            # Encode the frame as JPEG
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            # Stream the frame as a multipart response (MIME type: image/jpeg)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return '''
        <html>
            <head>
                <title>Object Detection Stream</title>
            </head>
            <body>
                <h1>Live Video Feed</h1>
                <img src="/video_feed" id="video-stream">
                <script src="https://cdnjs.cloudflare.com/ajax/libs/tensorflow/3.15.0/tf.min.js"></script>
                <script src="https://unpkg.com/@tensorflow-models/coco-ssd"></script>
                <script src="object_detection.js"></script>
            </body>
        </html>
    '''

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
