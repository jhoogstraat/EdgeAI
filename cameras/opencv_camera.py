from cameras.base_camera import BaseCamera
from cv2 import VideoCapture, imencode, cvtColor
from cv2 import CAP_V4L, COLOR_BGR2RGB, CAP_PROP_BUFFERSIZE, CAP_PROP_FRAME_WIDTH, CAP_PROP_FRAME_HEIGHT
from PIL import Image
from io import BytesIO

class OpenCVCamera(BaseCamera):
    def __init__(self, videoSource=0, dims=(640, 480)):
        w, h = dims

        self.vcap = VideoCapture(videoSource if not videoSource.isnumeric() else int(videoSource), CAP_V4L) # Using V4L backend, as the default (gstreamer) fails to change the resolution.
        if not self.vcap.isOpened():
            raise RuntimeError('Could not open camera')

        self.vcap.set(CAP_PROP_BUFFERSIZE, 1) # Minimize the frame buffer to always receive frames in realtime.
        self.vcap.set(CAP_PROP_FRAME_WIDTH, w)
        self.vcap.set(CAP_PROP_FRAME_HEIGHT, h)
        
        if (self.vcap.get(CAP_PROP_FRAME_WIDTH) != w or self.vcap.get(CAP_PROP_FRAME_HEIGHT) != h):
            raise RuntimeError('Resolution not supported')

        self.midX = int(w / 2)
        self.midY = int(h / 2)
        self.cropDim2 = int(min(w, h) / 2)
        
    @property
    def resolution(self):
        return { 
            'original': {
                'width': int(self.vcap.get(3)), 
                'height': int(self.vcap.get(4)) 
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

        crop = frame[self.midY-self.cropDim2 : self.midY+self.cropDim2, self.midX-self.cropDim2 : self.midX+self.cropDim2]
        return Image.fromarray(cvtColor(crop, COLOR_BGR2RGB)) # cv2 uses BGR instead of RGB by default.

    def encodeJPG(self, pilImage):
        jpg = BytesIO()
        pilImage.save(jpg, format='JPEG')
        return jpg.getvalue()