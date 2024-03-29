from pycoral.utils import edgetpu, dataset
from pycoral.adapters import common, detect
from .base_detector import ObjectDetector

class PyCoralDetector(ObjectDetector):

    def __init__(self, modelDir):
        super().__init__(name='PyCoralDetector')
        self.configure(modelDir)

    def configure(self, modelDir):
        # Workaround, if no edgetpu is available
        if not edgetpu.list_edge_tpus():
            print("No EdgeTPUs found. Using the CPU only...")
            from tflite_runtime.interpreter import Interpreter
            self.interpreter = Interpreter(modelDir + "/model.tflite")
        else:
            print("EdgeTPU found. Connecting to it via PyCoral...")
            from pycoral.utils.edgetpu import make_interpreter
            self.interpreter = make_interpreter(modelDir + "/model.tflite")
        self.interpreter.allocate_tensors()
        self.modelDir = modelDir
        self._inputSize = common.input_size(self.interpreter)
        self._labels = dataset.read_label_file(modelDir + "/labels.txt")

    @property
    def labels(self):
        return self._labels

    @property
    def inputSize(self):
        return {'width': int(self._inputSize[0]), 'height': int(self._inputSize[1])}

    def detect(self, pilImage):
        _, scale = common.set_resized_input(
            self.interpreter, pilImage.size, lambda size: pilImage.resize(size))
        self.interpreter.invoke()
        return detect.get_objects(self.interpreter, score_threshold=0.6, image_scale=scale)
