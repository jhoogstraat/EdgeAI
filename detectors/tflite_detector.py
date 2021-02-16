from detectors.azure.predict import TFLiteObjectDetection
from detectors.base_detector import ObjectDetector

class TFLiteDetector(ObjectDetector):

    def __init__(self, modelDir):
        with open(modelDir + "/labels.txt", 'r') as f:
            labels = [l.strip() for l in f.readlines()]

        self.interpreter = TFLiteObjectDetection(modelDir + "/model.tflite", labels)
        
    @property
    def labels(self):
        return self.interpreter.labels

    @property
    def inputSize(self):
        return { 'width': 0, 'height': 0 }
        # return { 'width': int(self._inputSize[0]), 'height': int(self._inputSize[1]) }

    def detect(self, pilImage):
        return self.interpreter.predict_image(pilImage)