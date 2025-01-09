class SparkImage:
    def __init__(self, rgb=None, depth=None):
        self._rgb = rgb
        self._depth = depth

    def get_rgb(self):
        return self._rgb

    def get_depth(self):
        return self._depth
