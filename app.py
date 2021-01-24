#!/usr/bin/env python3
import argparse
from cameras.opencv_camera import OpenCVCamera as Camera
# from detectors.azure_detector import TFLiteAzureObjectDetector as Detector
from detectors.pycoral_detector import PyCoralDetector as Detector
from set_checker import SetChecker
from flask_socketio import SocketIO
from flask import Flask, request, send_file
import json

app = Flask(__name__)
socketio = SocketIO(app)
setChecker = None

@app.route("/viewer")
def getViewer():
    return send_file("viewer.html")

@app.route("/status")
def status():
    return setChecker.status()

@app.route("/start")
def start():
    def emit(frame):
        socketio.emit('frame', frame)
        socketio.sleep(0)
    socketio.start_background_task(setChecker.start, callback=emit)
    return setChecker.status()

@app.route("/stop")
def stop():
    setChecker.stop()
    return setChecker.status()

@app.route("/set")
def configure():
    settings = request.json

    if 'set' in settings:
        setChecker.configureSet(settings['set'])

    return setChecker.status()

def main(args):
    global setChecker
    camera = Camera(videoSource=args.video)
    detector = Detector(args.model)
    setChecker = SetChecker(camera, detector)

    print('[INFO] Starting server at http://localhost:5000')
    socketio.run(app=app, host='0.0.0.0', port=5000)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Python Detection server using a webcam and models trained with AutoML.')
    
    parser.add_argument('-v', '--video', default="0", help='The source of the video (path)')
    parser.add_argument('-m', '--model', required=True, help='The directory containing a model.(pb|tflite) and a corresponding labels.txt, relative to the current dir.')

    args = parser.parse_args()
    main(args)

