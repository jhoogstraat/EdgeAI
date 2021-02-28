class Coordinator():
    """This module comprises the camera and detector, to check if a set is complete."""

    def __init__(self, camera, detector, usecase):
        self.camera = camera
        self.detector = detector
        self.usecase = usecase
    
        self.isRunning = False

    def configureUsecase(self, config):
        self.usecase.configure(config)
    
    def configureDetector(self, config):
        self.detector.configure(config['model'])

    def start(self, frameCallback, usecaseCallback):
        """Video streaming generator function."""

        # Start service only once.
        if self.isRunning:  
            return

        print("Starting Detection Service")

        self.camera.open()
        self.isRunning = True

        while self.isRunning:
            try:
                frame = self.camera.read()
                if (self.detector.name == 'TF2Detector'):
                    jpg, objects = self.detector.detect(frame)
                else:
                    jpg = self.camera.encodeJPG(frame)
                    objects = self.detector.detect(frame)
                self.usecase.run(jpg, objects, usecaseCallback)
                frameCallback({ 'frame': jpg, 'objects': [obj._asdict() for obj in objects] })
            except IOError:
                self.isRunning = False
                return

        # Close the camera when service is stopped.
        self.camera.close() 

    def stop(self):
        if (self.isRunning == True):
            print("Stopping Detection Service")
            self.isRunning = False

    def status(self):
        return {
            'isRunning': self.isRunning,
            'frameSize': self.camera.resolution,
            'inputSize': self.detector.inputSize,
            'labels': self.detector.labels,
            'usecase': self.usecase.status(),
            'detector': self.detector.name,
            'model': self.detector.modelDir
        }