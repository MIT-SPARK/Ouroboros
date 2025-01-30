"""Module containing interfaces for pose recovery."""

import abc
import logging
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from ouroboros.vlc_db.camera import PinholeCamera
from ouroboros.vlc_db.vlc_db import VlcDb
from ouroboros.vlc_db.vlc_image import VlcImage

# TODO(nathan) use actual type alias once we move beyond 3.8
# Matrix3d = np.ndarray[np.float64[3, 3]]
# Matrix4d = np.ndarray[np.float64[4, 4]]
Matrix3d = np.ndarray
Matrix4d = np.ndarray


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
        cls, cam: PinholeCamera, img: VlcImage, indices: Optional[np.ndarray] = None
    ):
        """Get undistorted geometry from keypoints."""
        depths = img.get_feature_depths()
        keypoints = img.keypoints
        if indices is not None:
            keypoints = keypoints[indices, :]
            depths = None if depths is None else depths[indices]

        bearings, points = get_bearings(cam.K, keypoints, depths)
        return cls(bearings, depths, points)


@dataclass
class PoseRecoveryResult:
    """Result for pose recovery."""

    query_T_match: Optional[Matrix4d] = None
    is_metric: bool = False
    inliers: Optional[List[int]] = None

    def __bool__(self):
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
        query_camera: Optional[PinholeCamera] = None,
        match_camera: Optional[PinholeCamera] = None,
    ):
        """Recover pose from two frames and correspondences."""
        if query.keypoints is None or match.keypoints is None:
            logging.error("Keypoints required for pose recovery!")
            return None

        if query_camera is None:
            query_camera = vlc_db.get_camera(query.metadata)
        if match_camera is None:
            match_camera = vlc_db.get_camera(match.metadata)

        # TODO(nathan) undistortion

        query_geometry = FeatureGeometry.from_image(
            query_camera, query, query_to_match[:, 0]
        )
        match_geometry = FeatureGeometry.from_image(
            match_camera, match, query_to_match[:, 1]
        )
        return self._recover_pose(query_geometry, match_geometry)

    @abc.abstractmethod
    def _recover_pose(
        self, bearings_q: np.ndarray, bearings_m: np.ndarray
    ) -> PoseRecoveryResult:
        pass  # pragma: no cover
