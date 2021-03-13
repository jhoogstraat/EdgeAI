import tensorflow as tf
import numpy as np
from PIL import Image
from .base_detector import ObjectDetector, Object, BBox

class TF2Detector(ObjectDetector):
    def __init__(self, modelDir):
        super().__init__(name='TF2Detector')
        self.configure(modelDir)

    def configure(self, modelDir):
        tf.keras.backend.clear_session()
        self.detectFn = tf.saved_model.load(modelDir).signatures[tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
        self.detect(Image.new('RGB', (480, 480))) # Warmup
        self.modelDir = modelDir

    @property
    def labels(self):
        return {
            '1': 'gelbes_auge',
            '2': 'gruener_stein',
            '3': 'gelbe_rutsche',
            '4': 'blauer_balken',
            '5': 'orangene_platte',
            '6': 'schwarze_stange'
        }

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