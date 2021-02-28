from PIL import Image
from .base_detector import Object, BBox
from .azure.tf.predict import TFObjectDetection
from .base_detector import ObjectDetector


class MSDetector(ObjectDetector):
    def __init__(self, modelDir):
        self.name = 'MSTFDetector'
        self.configure(modelDir)

    def configure(self, modelDir):
        with open(modelDir + "/labels.txt", 'r') as f:
            labels = [l.strip() for l in f.readlines()]
        self.interpreter = TFObjectDetection(modelDir + '/saved_model.pb', labels)
        self.detect(Image.new('RGB', (480, 480))) # Warmup
        self.modelDir = modelDir
        # self._inputSize = self.interpreter.interpreter.get_input_details()[0]['shape'][1:3]

    @property
    def labels(self):
        return dict(enumerate(self.interpreter.labels))

    @property
    def inputSize(self):
        return { 'width': 512, 'height': 512 }
        # return { 'width': int(self._inputSize[0]), 'height': int(self._inputSize[1]) }

    def detect(self, pilImage):
        objects = self.interpreter.predict_image(pilImage)
        return list(map(self.__mapObjectToPyCoralFormat, objects))

    def __mapObjectToPyCoralFormat(self, object):
        bbox = BBox(
            xmin=object['boundingBox']['left'] * 480,
            ymin=object['boundingBox']['top'] * 480,
            xmax=(object['boundingBox']['left'] +
                  object['boundingBox']['width']) * 480,
            ymax=(object['boundingBox']['top'] +
                  object['boundingBox']['height']) * 480
        )
        return Object(id=object['tagId'], score=object['probability'], bbox=bbox)
