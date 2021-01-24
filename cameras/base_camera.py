class BaseCamera(object):

    @property
    def resolution(self):
        """Provides the camera resolution."""
        raise RuntimeError('Must be implemented by subclasses.')

    def read(self):
        """Retrieves a frame from the camera as a (numpy.ndarray).
        """
        raise RuntimeError('Must be implemented by subclasses.')
    
    def encodeJPG(self, frame):
        """Encode a image tensor (numpy.ndarray) to jpg.

        Args:
            frame (numpy.ndarray): The frame to be encoded.

        Returns:
            A memory-buffer containing the JPG image.
        """
        raise RuntimeError('Must be implemented by subclasses.')
