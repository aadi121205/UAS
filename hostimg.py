import time
import threading
import cv2
import threading
from flask import Response, Flask
import time
import math

# Create the Flask object for the application
app = Flask(__name__)

app2 = Flask(__name__)

def serveJson(jason_data):
    # Return the JSON data as a response
    return Response(jason_data, mimetype="application/json")

def streamFrame(frame):

    if frame is None:
        return "Failed to capture frame"

    # Encode the frame as PNG
    return_key, encoded_image = cv2.imencode(".png", frame)
    if not return_key:
        return "Failed to encode frame"

    # Output image as a byte array
    return Response(b'--frame\r\n' b'Content-Type: image/png\r\n\r\n' + bytearray(encoded_image) + b'\r\n', mimetype="multipart/x-mixed-replace; boundary=frame")

# check to see if this is the main thread of execution
if __name__ == "__main__":
    with open("data.json", "r") as file:
        json_data = file.read()
    app.run(host="0.0.0.0", port="5555")
    # Create a VideoCapture object
    frame = cv2.imread("host.png")
    # Start the thread that will serve the frames
    threading.Thread(target=streamFrame, args=(frame,)).start()
    # Start the Flask app
    threading.Thread(target=app.run, args=(json_data,)).start()
    # Wait for the threads to finish





