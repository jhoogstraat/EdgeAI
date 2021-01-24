import threading
from time import time
from collections import Counter

class SetChecker():
    """This module comprises the camera and detector, to check if a set is complete."""

    def __init__(self, camera, detector, initialSet={}):
        self.camera = camera
        self.detector = detector

        # Config
        self.set = initialSet # The complete set.
        self.minObjectCount = 2 # Minimum  object count to start set check.
        self.requiredPercentage = 0.6 # Percentage of frames required to count object as detected. (0.5 is minimum)
        self.recordDuration = 2 # Record duration in seconds.
        self.timeoutDuration = 1 # Timeout in seconds.
        
        # State
        self.isRunning = False

    def configureSet(self, _set):
        self.set = { int(key): value for key, value in _set.items() }

    def start(self, callback):
        """Video streaming generator function."""

        if (self.isRunning): # Start service only once.
            return

        print("Starting Detection Service")

        # State
        self.isRunning = True
        self.detectionTime = None
        self.timeoutTime = None
        self.objectStore = None

        while (self.isRunning):
            try:
                frame = self.camera.read()
            except IOError:
                self.isRunning = False
                return

            objects = self.detector.detect(frame)
            
            # Reset timeout if expired.
            if (self.timeoutTime is not None and secondsSince(self.timeoutTime) >= self.timeoutDuration):
                self.detectionTime = None
                self.timeoutTime = None

            if (self.timeoutTime is None): # Run set checker if not in timeout.
                if (self.detectionTime is None): # Start set checker if not already running.
                    if (len(objects) >= self.minObjectCount):
                        print('\n---------START---------')
                        print('Gesucht: {:}'.format(self.set))
                        self.detectionTime = time()
                        self.objectStore = [objects]

                elif (secondsSince(self.detectionTime) <= self.recordDuration): # Record for the configured duration.
                    self.objectStore.append(objects)

                else: # Evaluate results after recording is done.
                    # labels = self.detector.labels
                    objectCount = Counter([obj.id for frame in self.objectStore for obj in frame])
                    print(objectCount)
                    frameCount = len(self.objectStore)
                    minRequiredObjectOccurence = round(self.requiredPercentage * frameCount)

                    set_complete = True
                    if (len(objectCount) != len(self.set)):
                        set_complete = False
                        print("Detected additional Objects!")
                    else:
                        for className, classCount in self.set.items():
                            requiredDetectionCount = minRequiredObjectOccurence * classCount
                            if (objectCount[className] < requiredDetectionCount):
                                set_complete = False
                    
                    print('Analysierte Bilder: {:>4}'.format(frameCount))
                    print('Benötigt ({:}%):{:>7}'.format(self.requiredPercentage * 100, minRequiredObjectOccurence))
                    print('Erkannt:')
                    
                    for objClass, count in objectCount.items():
                        print('{:>24}/{:}x {:}'.format(count, minRequiredObjectOccurence * (self.set.get(objClass) or 0), objClass))
                    
                    print("SET VOLLSTÄNDIG" if set_complete else "SET FEHLERHAFT")

                    self.timeoutTime = time() # Start timeout.
                    self.objectStore = None # Reset store.
                    
            frame = {
                'frame': self.camera.encodeJPG(frame), 
                'objects': [obj._asdict() for obj in objects]
            }

            callback(frame)

    def stop(self):
        if (self.isRunning == True):
            print("Stopping Detection Service")
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