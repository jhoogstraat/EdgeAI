from cameras.base_camera import BaseCamera
from cv2 import VideoCapture, cvtColor
from cv2 import CAP_V4L, COLOR_BGR2RGB, CAP_PROP_BUFFERSIZE, CAP_PROP_FRAME_WIDTH, CAP_PROP_FRAME_HEIGHT
from PIL import Image
from io import BytesIO


class OpenCVCamera(BaseCamera):
    def __init__(self, videoSource, dims=(640, 480)):
        w, h = dims
        self.w = w
        self.h = h
        self.midX = int(w / 2)
        self.midY = int(h / 2)
        self.cropDim2 = int(min(w, h) / 2)

        self.videoSource = videoSource
        self.vcap = VideoCapture()

    def open(self):
        if not self.vcap.isOpened() and not self.vcap.open(self.videoSource, CAP_V4L):
            raise RuntimeError('Could not open camera')

        # Minimize the frame buffer to always receive frames in realtime.
        self.vcap.set(CAP_PROP_BUFFERSIZE, 1)
        self.vcap.set(CAP_PROP_FRAME_WIDTH, self.w)
        self.vcap.set(CAP_PROP_FRAME_HEIGHT, self.h)

        if (self.vcap.get(CAP_PROP_FRAME_WIDTH) != self.w or self.vcap.get(CAP_PROP_FRAME_HEIGHT) != self.h):
            raise RuntimeError('Resolution not supported')
    
    def close(self):
        self.vcap.release()

    @property
    def resolution(self):
        return {
            'original': {
                'width': self.w,
                'height': self.h
            },
            'crop': {
                'width': self.cropDim2 * 2,
                'height': self.cropDim2 * 2
            }
        }

    def read(self):
        (success, frame) = self.vcap.read()
        if (success == False):
            raise IOError('Frame could not be read from source')

        crop = frame[self.midY-self.cropDim2: self.midY+self.cropDim2,
                     self.midX-self.cropDim2: self.midX+self.cropDim2]
        # cv2 uses BGR instead of RGB by default.
        return Image.fromarray(cvtColor(crop, COLOR_BGR2RGB))

    def encodeJPG(self, pilImage):
        jpg = BytesIO()
        pilImage.save(jpg, format='JPEG')
        return jpg.getvalue()
