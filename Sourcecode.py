from flask import Flask, render_template, Response, request, jsonify
from flask_socketio import SocketIO, emit
import cv2
import threading

app = Flask(__name__)
socketio = SocketIO(app)
camera = cv2.VideoCapture(0)
if not camera.isOpened():
    print("Error: Could not open video source")
motion_detection = False
frame_rate = 30
resolution = (640, 480)

def set_camera_settings():
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
    camera.set(cv2.CAP_PROP_FPS, frame_rate)

def generate_frames():
    global motion_detection
    prev_frame = None

    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            if motion_detection:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray = cv2.GaussianBlur(gray, (21, 21), 0)

                if prev_frame is None:
                    prev_frame = gray
                    continue

                frame_delta = cv2.absdiff(prev_frame, gray)
                thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
                thresh = cv2.dilate(thresh, None, iterations=2)
                contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                for contour in contours:
                    if cv2.contourArea(contour) < 500:
                        continue
                    (x, y, w, h) = cv2.boundingRect(contour)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                prev_frame = gray

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@socketio.on('toggle_motion')
def handle_toggle_motion(data):
    global motion_detection
    motion_detection = data['status']

@socketio.on('change_settings')
def handle_change_settings(data):
    global frame_rate, resolution
    frame_rate = data['frame_rate']
    resolution = (data['width'], data['height'])
    set_camera_settings()

if __name__ == '__main__':
    set_camera_settings()
    socketio.run(app, host='0.0.0.0', port=5000)
