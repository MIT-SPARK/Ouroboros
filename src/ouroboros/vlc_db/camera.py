"""Class representing a camera for a particular image."""

from dataclasses import dataclass

import numpy as np


@dataclass
class PinholeCamera:
    """Class representing a pinhole camera."""

    # NOTE(nathan) a unit camera seems like a sane default
    fx: float = 1.0
    fy: float = 1.0
    cx: float = 0.0
    cy: float = 0.0

    @property
    def K(self):
        """Get the (undistorted) camera matrix for the camera."""
        mat = np.eye(3)
        mat[0, 0] = self.fx
        mat[1, 1] = self.fy
        mat[0, 2] = self.cx
        mat[1, 2] = self.cy
        return mat


@dataclass
class VlcCamera:
    """Class containing database information about a camera."""

    camera: PinholeCamera
