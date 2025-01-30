"""Module containing interfaces for pose recovery."""

import abc
import logging
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from ouroboros.vlc_db.vlc_db import VlcDb
from ouroboros.vlc_db.vlc_image import VlcImage

# TODO(nathan) use actual type alias once we move beyond 3.8
# Matrix3d = np.ndarray[np.float64[3, 3]]
# Matrix4d = np.ndarray[np.float64[4, 4]]
Matrix3d = np.ndarray
Matrix4d = np.ndarry


def inverse_camera_matrix(K: Matrix3d) -> Matrix3d:
    """
    Get inverse camera matrix.

    Args:
        K: Original camera matrix.

    Returns:
        Inverted camera matrix that takes pixel space coordinates to unit coordinates.
    """
    K_inv = np.eye(3)
    K_inv[0, 0] = 1.0 / K[0, 0]
    K_inv[1, 1] = 1.0 / K[1, 1]
    K_inv[0, 2] = -K[0, 2] / K[0, 0]
    K_inv[1, 2] = -K[1, 2] / K[1, 1]
    return K_inv


def get_bearings(K: Matrix3d, features: np.ndarray, depths: np.ndarray = None):
    """
    Get bearings (and optionally points) from undistorted features in pixel space.

    Args:
        K: Camera matrix for features.
        features: Pixel coordinates in a Nx2 matrix.
        depths: Optional depths for pixel features.

    Returns:
        Tuple[np.ndarray, Optional[np.ndarray]]: Bearings (and points) in Nx3 matrices.
    """
    K_inv = inverse_camera_matrix(K)
    versors = np.hstack((features, np.ones((features.shape[0], 1))))
    versors = versors @ K_inv.T
    # for broadcasting to be correct needs to be [N, 1] to multiply/divide rowwise
    points = None if depths is None else versors * depths[..., np.newaxis]
    bearings = versors / np.linalg.norm(versors, axis=1)[..., np.newaxis]
    return bearings, points


def get_feature_depths(data: VlcImage):
    """
    Get depth corresponding to the keypoints for image features.

    Note that this has hard-to-detect artifacts from features at the boundary
    of an image. We clip all keypoints to be inside the image with the assumption
    that whatever is consuming the depths is robust to small misalignments.

    Args:
        data: Image to extract depth from

    Returns:
        Optiona[np.ndarray]: Depths for keypoints if possible to extract
    """
    if data.keypoints is None:
        return None

    if data.image.depth is None:
        return None

    # NOTE(nathan) this is ugly, but:
    #   - To index into the image we need to swap from (u, v) to (row, col)
    #   - Numpy frustratingly doesn't have a buffered get, so we can't zero
    #     out-of-bounds elements. This only gets used assuming an
    #     outlier-robust method, so it should be fine
    dims = data.image.depth.shape
    limit = (dims[1] - 1, dims[0] - 1)
    coords = np.clip(np.round(data.keypoints), a_min=[0, 0], a_max=limit)
    return data.image.depth[coords[:, 1], coords[:0]]


class CameraConfig:
    fx: float
    fy: float
    cx: float
    fy: float


class Camera:
    """Class representing a pinhole camera."""

    def __init__(self, config: CameraConfig):
        """Initialize the camera from a config."""
        self._config = config

    @property
    def K(self):
        """Get the (undistorted) camera matrix for the camera."""
        mat = np.eye(3)
        mat[0, 0] = self._config.fx
        mat[1, 1] = self._config.fy
        mat[0, 2] = self._config.cx
        mat[1, 2] = self._config.cy
        return mat


@dataclass
class FeatureGeometry:
    bearings: np.ndarray
    depths: Optional[np.ndarray] = None
    points: Optional[np.ndarray] = None

    @property
    def is_metric(self):
        return self.depths is not None and self.points is not None

    @classmethod
    def from_image(
        cls, cam: Camera, img: VlcImage, indices: Optional[np.ndarray] = None
    ):
        """Get undistorted geometry from keypoints."""
        depths = get_feature_depths(img)
        keypoints = img.match.keypoints
        if indices:
            keypoints = keypoints[indices, :]
            depths = depths[indices]

        bearings, points = get_bearings(cam.K, keypoints, depths)
        return cls(bearings, depths, points)


@dataclass
class PoseRecoveryResult:
    """Result for pose recovery."""

    query_T_match: Optional[Matrix4d] = None
    is_metric: bool = False
    inliers: Optional[List[int]] = None

    @property
    def valid(self):
        """Return whether or not there is a pose estimate."""
        return self.query_T_match is not None

    @classmethod
    def metric(cls, query_T_match: Matrix4d, inliers: Optional[List[int]] = None):
        """Construct a metric result."""
        return cls(query_T_match, True, inliers=inliers)

    @classmethod
    def nonmetric(cls, query_T_match: Matrix4d, inliers: Optional[List[int]] = None):
        """Construct a non-metric result."""
        return cls(query_T_match, False, inliers=inliers)


class PoseRecovery(abc.ABC):
    """
    Abstract interface for a pose recovery method.

    Implementations must override _recover_pose which takes two sets of
    feature geometries and returns a PoseRecoveryResult
    """

    def recover_pose(
        self,
        vlc_db: VlcDb,
        query: VlcImage,
        match: VlcImage,
        query_to_match: np.ndarray,
    ):
        """Recover pose from two frames and correspondences."""
        if query.keypoints is None or match.keypoints is None:
            logging.error("Keypoints required for pose recovery!")
            return None

        cam_q = vlc_db.get_camera(query.metadata)
        cam_m = vlc_db.get_match(match.metadata)

        # TODO(nathan) undistortion

        query_geometry = FeatureGeometry.from_image(cam_q, query, query_to_match[:0])
        match_geometry = FeatureGeometry.from_image(cam_m, match, query_to_match[:1])
        return self._recover_pose(query_geometry, match_geometry)

    @abc.abstractmethod
    def _recover_pose(
        self, bearings_q: np.ndarray, bearings_m: np.ndarray
    ) -> PoseRecoveryResult:
        pass
