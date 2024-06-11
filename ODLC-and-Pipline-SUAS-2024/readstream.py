import cv2
from flask import Flask, Response

app = Flask(__name__)

vc = cv2.VideoCapture('rtmp://your_rtmp_stream')

def gen():
    while True:
        r, frame = vc.read()
        if not r:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)