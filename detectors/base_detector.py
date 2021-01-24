class ObjectDetector(object):
    def labels(self):
        """Produces a compact, json-usable list of labels.

        Returns:
            List of (int) which maps label id to class name.
        """
        raise RuntimeError('Must be implemented by subclasses.')

    def inputSize(self):
        """Produces a compact, json-usable dict of the model's input size.

        Returns:
            dict with keys 'w' (width) and 'h' (height) mapping to (int) sizes.
        """
        raise RuntimeError('Must be implemented by subclasses.')
    
    def detect(self, pilImage):
        """Single frame detection function.

        Returns:
            list of namedtuples in shape of ('Object', ['id', 'score', 'bbox']).
        """
        raise RuntimeError('Must be implemented by subclasses.')