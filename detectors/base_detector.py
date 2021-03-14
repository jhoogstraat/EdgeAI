import collections

Object = collections.namedtuple('Object', ['id', 'score', 'bbox'])
BBox = collections.namedtuple('BBox', ['xmin', 'ymin', 'xmax', 'ymax'])


class ObjectDetector(object):
    def __init__(self, name):
        self.name = name

    def labels(self):
        """Produces a compact, json-usable list of labels.

        Returns:
            Dict of (int, string) which maps label id to description.
        """
        raise NotImplementedError

    def inputSize(self):
        """Produces a compact, json-usable dict of the model's input size.

        Returns:
            Dict with keys 'w' (width) and 'h' (height) mapping to (int) sizes.
        """
        raise NotImplementedError

    def detect(self, pilImage):
        """Single frame detection function.

        Returns:
            list of namedtuples in shape of ('Object', ['id', 'score', 'bbox']).
        """
        raise NotImplementedError
