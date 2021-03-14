#!/usr/bin/env python3
import argparse
from coordinator import Coordinator
from cameras.opencv_camera import OpenCVCamera as Camera
from flask_socketio import SocketIO
from flask import Flask, request, send_file
from usecases.check_set import CheckSetUseCase


app = Flask(__name__)
socketio = SocketIO(app)
coordinator = None


@app.route("/viewer")
def getViewer():
    return send_file("viewer.html")


@app.route("/status")
def status():
    return coordinator.status()


@app.route("/start", methods=['POST'])
def start():
    def emitFrame(frame):
        socketio.emit('frame', frame)
        socketio.sleep(0)

    def emitUsecase(featureSet):
        socketio.emit('set', featureSet)
        socketio.sleep(0)

    socketio.start_background_task(
        coordinator.start, frameCallback=emitFrame, usecaseCallback=emitUsecase)
    return coordinator.status()


@app.route("/stop", methods=['POST'])
def stop():
    coordinator.stop()
    return coordinator.status()

@app.route("/configureDetector", methods=['POST'])
def configureDetector():
    coordinator.configureDetector(request.json)
    return coordinator.status()


@app.route("/configureUsecase", methods=['POST'])
def configureUsecase():
    coordinator.configureUsecase(request.json)
    return coordinator.status()


def main(args):
    global coordinator

    if args.detector == 'pycoral':
        print("Loading PyCoral Detector")
        from detectors.pycoral_detector import PyCoralDetector as Detector
    elif args.detector == 'tf2':
        print("Loading TF2 Detector")
        from detectors.tf2_detector import TF2Detector as Detector
    elif args.detector == 'ms-tflite':
        print("Loading MS TFLite-Detector")
        from detectors.ms_detector import MSTFLiteDetector as Detector
    elif args.detector == 'ms-tf2':
        print("Loading MS TF-Detector")
        from detectors.ms_detector import MSTFDetector as Detector

    camera = Camera(videoSource=args.video)
    detector = Detector(args.model)
    usecase = CheckSetUseCase()
    coordinator = Coordinator(camera, detector, usecase)

    print('[INFO] Starting server at http://localhost:5000')
    socketio.run(app=app, host='0.0.0.0', port=5000)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Python Detection server using a webcam and models trained with AutoML.')

    parser.add_argument('-v', '--video', default='/dev/video0',
                        help='The source of the video (path)')
    parser.add_argument('-m', '--model', required=True,
                        help='The directory containing a model.(pb|tflite) and a corresponding labels.txt, relative to the current dir.')
    parser.add_argument('-d', '--detector', choices=['pycoral', 'ms-tflite', 'ms-tf2', 'tf2'], default="pycoral",
                        help='The detector to be used. Has to be compatible to the model.')
    args = parser.parse_args()
    main(args)
