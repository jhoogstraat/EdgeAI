import tensorflow as tf
import numpy as np
from PIL import Image
import collections

Object = collections.namedtuple('Object', ['id', 'score', 'bbox'])
BBox = collections.namedtuple('BBox', ['xmin', 'ymin', 'xmax', 'ymax'])

class TF2Detector():
    def __init__(self, modelDir):
        tf.keras.backend.clear_session()
        self.detectFn = tf.saved_model.load(modelDir).signatures['serving_default']

    @property
    def labels(self):
        return {}

    @property
    def inputSize(self):
        return { 'width': 0, 'height': 0 }

    def detect(self, pilImage):
        '''
            npImage has to be (x, x, 3) and dtype=uint8
        '''
        npImage = np.array(pilImage)
        jpg = tf.io.encode_jpeg(npImage)
        batchInput = tf.expand_dims(jpg, axis=0)
        output = self.detectFn(image_bytes=batchInput, key=tf.expand_dims(tf.convert_to_tensor("img"), 0))
        return (jpg.numpy(), self.getObjects(output, imageSize=npImage.shape[0]))

    def getObjects(self, output, imageSize, scoreThreshold=0.7):
        count = int(output['num_detections'][0])
        boxes = output['detection_boxes'][0]
        classes = output['detection_classes'][0]
        scores = output['detection_scores'][0]

        def make(i):
            ymin, xmin, ymax, xmax = boxes[i].numpy()
            bbox = BBox(xmin=float(imageSize * xmin),
                        ymin=float(imageSize * ymin),
                        xmax=float(imageSize * xmax),
                        ymax=float(imageSize * ymax))
            return Object(id=int(classes[i]), score=float(scores[i]), bbox=bbox)
        
        return [make(i) for i in range(count) if scores[i] >= scoreThreshold]