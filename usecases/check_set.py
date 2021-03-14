from collections import Counter
import csv
from time import time, ctime


class CheckSetUseCase():

    def __init__(self, objects):
        # Config
        self.set = {id: 0 for id in objects.keys()}  # The complete set.
        self.minObjects = 2  # Minimum  object count to start set check.
        self.recordDuration = 2  # Record duration in seconds.
        self.timeoutDuration = 1  # Timeout in seconds.

        # State
        self.stats = open('stats.csv', mode='a')
        self.csvWriter = csv.writer(
            self.stats, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        self.detectionTime = None
        self.timeoutTime = None
        self.objectStore = None
        self.referenceFrame = None
        self.setId = 0

    def configure(self, config):
        if 'set' in config:
            self.set = {int(key): value for key,
                        value in config['set'].items()}
        if 'minObjects' in config:
            self.minObjects = config['minObjects']
        if 'recordDuration' in config:
            self.recordDuration = config['recordDuration']
        if 'timeoutDuration' in config:
            self.timeoutDuration = config['timeoutDuration']

    def status(self):
        return {'set': self.set}

    def run(self, frame, objects, callback):
        # Skip if in timeout and reset if timeout expired.
        if self.timeoutTime is not None:
            if secondsSince(self.timeoutTime) >= self.timeoutDuration:
                self.detectionTime = None
                self.timeoutTime = None
            return

        # Start set checker if not running already.
        if (self.detectionTime is None):
            if (len(objects) >= self.minObjects):
                self.detectionTime = time()
                self.objectStore = [objects]

        # Record for the configured duration.
        elif secondsSince(self.detectionTime) <= self.recordDuration:
            self.objectStore.append(objects)
            if self.referenceFrame is None and secondsSince(self.detectionTime) >= self.recordDuration / 2:
                self.referenceFrame = frame

        # Evaluate results after recording is done.
        else:
            setDetected = Counter(
                [obj.id for frame in self.objectStore for obj in frame])
            frameCount = len(self.objectStore)
            print(self.set, setDetected)
            inTargetRange = {}
            for id, target in self.set.items():
                upperBound = frameCount * target
                lowerBound = max(0, frameCount * (target - 1) + 1)

                inTargetRange[id] = lowerBound <= setDetected[id] <= upperBound

                self.csvWriter.writerow([0,
                                         0,
                                         0,
                                         'device',
                                         'model',
                                         len(self.set),
                                         ctime(),
                                         list(self.set.items()),
                                         'gt', frameCount,
                                         list(sorted(setDetected.items())),
                                         ])

            featureSet = {
                'id': self.setId,
                'requestedSet': self.set,
                'detectedSet': setDetected,
                'ok': all(inTargetRange.values()),
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
