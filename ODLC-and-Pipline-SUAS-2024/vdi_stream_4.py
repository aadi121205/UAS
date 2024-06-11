import cv2
import threading
from flask import Response, Flask
import datetime

# Dictionary to store the frames of each camera
camera_frames = {}

# Use locks for thread-safe access to camera frames
camera_locks = {}

# Create the Flask object for the application
app = Flask(__name__)

# Define global variables for video writing
video_writers = {}


def captureFrames(camera_index):
    global camera_captures, camera_frames, camera_locks, video_writers

    # Video capturing from OpenCV
    capture = cv2.VideoCapture(camera_index)
    if not capture.isOpened():
        print(f"Failed to open camera {camera_index}")
        return

    # Define the video writer for storing the frames
    output_file = f"yedekho_h10_{camera_index}.avi"  # Customize the filename as needed
    output_codec = cv2.VideoWriter_fourcc(*'mp4v')
    output_fps = 30.0
    output_size = (320, 240)
    video_writer = cv2.VideoWriter(output_file, output_codec, output_fps, output_size)

    # Store the video writer in the global dictionary
    video_writers[camera_index] = video_writer
    n = 0

    while True and capture.isOpened():
        return_key, frame = capture.read()
        n = n + 1

        if not return_key:
            break

        # Store the frame in the dictionary, with thread-safe access
        with camera_locks[camera_index]:
            camera_frames[camera_index] = frame.copy()

        # Write the frame to the video file
        video_writer.write(frame)

        key = cv2.waitKey(30) & 0xff
        if key == 27:
            break

    capture.release()
    video_writer.release()

def encodeFrame(camera_index):
    global camera_frames, camera_locks

    while True:
        # Acquire lock to access the camera frame
        with camera_locks[camera_index]:
            if camera_frames[camera_index] is None:
                continue

            # Encode the frame
            
            return_key, encoded_image = cv2.imencode(".png", camera_frames[camera_index])
            if not return_key:
                continue

        # Output image as a byte array
        yield (b'--frame\r\n' b'Content-Type: image/png\r\n\r\n' + bytearray(encoded_image) + b'\r\n')

@app.route("/camera/<int:camera_index>")
def streamFrames(camera_index):
    return Response(encodeFrame(camera_index), mimetype="multipart/x-mixed-replace; boundary=frame")

# check to see if this is the main thread of execution
if __name__ == "__main__":
    # Create threads and locks for each camera
    camera_index = 0
    camera_frames[camera_index] = None
    camera_locks[camera_index] = threading.Lock()
    process_thread = threading.Thread(target=captureFrames, args=(camera_index,))
    process_thread.daemon = True
    process_thread.start()
    app.run(host="0.0.0.0", port="5555")

