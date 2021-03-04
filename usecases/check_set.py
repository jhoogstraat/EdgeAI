from collections import Counter
import csv
from time import time, ctime


class CheckSetUseCase():

    def __init__(self):
        # Config
        self.set = {}  # The complete set.
        self.minObjectCount = 2  # Minimum  object count to start set check.
        # Percentage of frames required to contain the object.
        self.requiredMinPercentage = 0.3
        self.recordDuration = 2  # Record duration in seconds.
        self.timeoutDuration = 1  # Timeout in seconds.

        # State
        self.detectionTime = None
        self.timeoutTime = None
        self.objectStore = None
        self.referenceFrame = None
        self.setId = 0
        self.tripCounter = 1

    def configure(self, config):
        if 'set' in config:
            self.set = {int(key): value for key,
                        value in config['set'].items()}

        if 'minPercentage' in config:
            self.requiredMinPercentage = config['minPercentage']

    def status(self):
        return {'set': self.set, 'minPercentage': self.requiredMinPercentage }

    def run(self, frame, objects, callback):
        # Reset timeout if expired.
        if self.timeoutTime is not None and secondsSince(self.timeoutTime) >= self.timeoutDuration:
            self.detectionTime = None
            self.timeoutTime = None

        # Run set checker if not in timeout.
        if (self.timeoutTime is None):
            # Start set checker if not already running.
            if (self.detectionTime is None):
                if (len(objects) >= self.minObjectCount):
                    self.detectionTime = time()
                    self.objectStore = [objects]

            # Record for the configured duration.
            elif secondsSince(self.detectionTime) <= self.recordDuration:
                self.objectStore.append(objects)
                if self.referenceFrame == None and secondsSince(self.detectionTime) >= self.recordDuration / 2:
                    self.referenceFrame = frame

            else:  # Evaluate results after recording is done.
                objectCount = Counter(
                    [obj.id for frame in self.objectStore for obj in frame])
                frameCount = len(self.objectStore)
                minRequiredObjectOccurence = round(
                    self.requiredMinPercentage * frameCount)
                maxRequiredObjectOccurence = frameCount

                isComplete = True
                if len(objectCount) != len(self.set):
                    # Check if only required objects were detected.
                    isComplete = False
                else:
                    for classId, classCount in self.set.items():
                        minDetections = minRequiredObjectOccurence * classCount
                        maxDetections = maxRequiredObjectOccurence * classCount
                        if not minDetections <= objectCount[classId] <= maxDetections:
                            isComplete = False

                with open('pi.csv', mode='a') as sets_file:
                    writer = csv.writer(
                        sets_file, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                    writer.writerow([0, 0, 0, 'device', 'model', len(list(self.set.items())), ctime(), list(self.set.items()), 'gt', frameCount, list(sorted(objectCount.items()))])
                
                self.tripCounter += 1

                featureSet = {
                    'id': self.setId,
                    'requestedFeatures': self.set,
                    'detectedFeatures': objectCount,
                    'isComplete': isComplete,
                    'referenceFrame': self.referenceFrame,
                    'timestamp': time(),
                }

                callback(featureSet)

                self.timeoutTime = time()  # Start timeout.
                self.objectStore = None  # Reset store.
                self.referenceFrame = None
                self.setId += 1


# Utils
def secondsSince(firstTime):
    return time() - firstTime
