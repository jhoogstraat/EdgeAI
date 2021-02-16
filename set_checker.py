import threading
from time import time, ctime
from collections import Counter
import csv

class SetChecker():
    """This module comprises the camera and detector, to check if a set is complete."""

    def __init__(self, camera, detector, initialSet={}):
        self.camera = camera
        self.detector = detector

        # Config
        self.set = initialSet # The complete set.
        self.minObjectCount = 2 # Minimum  object count to start set check.
        self.requiredPercentage = 0.3 # Percentage of frames required to count object as detected. (0.5 is minimum)
        self.recordDuration = 2 # Record duration in seconds.
        self.timeoutDuration = 1 # Timeout in seconds.
        self.log = open("log.csv", "a")
        self.logger = csv.writer(self.log, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

        # State
        self.isRunning = False

    def configureSet(self, _set):
        self.set = { int(key): value for key, value in _set.items() }

    def configurePercentage(self, percentage):
        self.requiredPercentage = percentage

    def start(self, frameCallback, setCallback):
        """Video streaming generator function."""

        if (self.isRunning): # Start service only once.
            return

        print("Starting Detection Service")

        # State
        self.isRunning = True
        self.detectionTime = None
        self.timeoutTime = None
        self.objectStore = None
        self.referenceFrame = None

        while (self.isRunning):
            try:
                frame = self.camera.read()
                jpg = self.camera.encodeJPG(frame)
            except IOError:
                self.isRunning = False
                return

            objects = self.detector.detect(frame)

            # Reset timeout if expired.
            if (self.timeoutTime is not None and secondsSince(self.timeoutTime) >= self.timeoutDuration):
                self.detectionTime = None
                self.timeoutTime = None

            # Run set checker if not in timeout.
            if (self.timeoutTime is None): 

                # Start set checker if not already running.
                if (self.detectionTime is None): 
                    if (len(objects) >= self.minObjectCount):
                        self.detectionTime = time()
                        self.objectStore = [objects]
                        self.referenceFrame = jpg

                # Record for the configured duration when started.
                elif (secondsSince(self.detectionTime) <= self.recordDuration): 
                    self.objectStore.append(objects)

                # Evaluate results after recording is done.
                else: 
                    objectCount = Counter([obj.id for frame in self.objectStore for obj in frame])
                    frameCount = len(self.objectStore)
                    minRequiredObjectOccurence = round(self.requiredPercentage * frameCount)

                    set_complete = True
                    if (len(objectCount) != len(self.set)):
                        set_complete = False
                    else:
                        for classId, classCount in self.set.items():
                            requiredDetectionCount = minRequiredObjectOccurence * classCount
                            if (objectCount[classId] < requiredDetectionCount):
                                set_complete = False
                
                    self.logger.writerow([
                        ctime(), 
                        list(self.set.items()), 
                        self.requiredPercentage, 
                        frameCount, 
                        minRequiredObjectOccurence, 
                        list(sorted(objectCount.items())), 
                        set_complete
                    ])

                    featureSet = {
                        'requestedFeatures': self.set,
                        'detectedFeatures': objectCount.items(),
                        'isComplete': set_complete,
                        'referenceFrame': self.referenceFrame,
                        'timestamp' : time(),
                    }

                    setCallback(featureSet)

                    self.timeoutTime = time() # Start timeout.
                    self.objectStore = None # Reset store.
                    self.referenceFrame = None
                    
            frame = {
                'frame': jpg, 
                'objects': [obj._asdict() for obj in objects]
            }

            frameCallback(frame)

    def stop(self):
        if (self.isRunning == True):
            print("Stopping Detection Service")
            self.log.close()
            self.isRunning = False

    def status(self):
        return {
            'frameSize': self.camera.resolution,
            'inputSize': self.detector.inputSize,
            'labels': self.detector.labels,
            'set': self.set,
            'isRunning': self.isRunning
        }

## Utils
def secondsSince(firstTime):
    return time() - firstTime