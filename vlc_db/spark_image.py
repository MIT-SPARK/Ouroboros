from dataclasses import dataclass

import numpy as np


@dataclass
class SparkImage:
    rgb: np.ndarray = None
    depth: np.ndarray = None

    def __init__(self, rgb, depth):
        self.rgb=rgb
        self.depth=depth

    def __repr__(self):
        s = ""
        if self.rgb is not None:
            shape = self.rgb.shape
            s += f"RGB image {shape[0]} x {shape[1]}\n"
        if self.depth is not None:
            shape = self.depth.shape
            s += f"Depth image {shape[0]} x {shape[1]}\n"

        return s
