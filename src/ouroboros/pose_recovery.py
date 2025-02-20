"""Module containing interfaces for pose recovery."""

import abc
import logging
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from ouroboros.vlc_db.camera import PinholeCamera
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
    """Associated feature geometry."""

    bearings: np.ndarray
    depths: Optional[np.ndarray] = None
    points: Optional[np.ndarray] = None

    @property
    def is_metric(self):
        """Whether or not the features have depth."""
        return self.depths is not None and self.points is not None

    @classmethod
    def from_image(
        cls, cam: PinholeCamera, img: VlcImage, indices: Optional[np.ndarray] = None
    ):
        """Get undistorted geometry from keypoints."""
        if not img.keypoint_depths:
            img.compute_feature_depths()

        depths = img.keypoint_depths
        keypoints = img.keypoints
        if indices is not None:
            keypoints = keypoints[indices, :]
            depths = None if depths is None else depths[indices]

        bearings, points = get_bearings(cam.K, keypoints, depths)
        return cls(bearings, depths, points)


@dataclass
class PoseRecoveryResult:
    """Result for pose recovery."""

    match_T_query: Optional[Matrix4d] = None
    is_metric: bool = False
    inliers: Optional[List[int]] = None

    def __bool__(self):
        """Return whether or not there is a pose estimate."""
        return self.match_T_query is not None

    @classmethod
    def metric(cls, match_T_query: Matrix4d, inliers: Optional[List[int]] = None):
        """Construct a metric result."""
        return cls(match_T_query, True, inliers=inliers)

    @classmethod
    def nonmetric(cls, match_T_query: Matrix4d, inliers: Optional[List[int]] = None):
        """Construct a non-metric result."""
        return cls(match_T_query, False, inliers=inliers)


class PoseRecovery(abc.ABC):
    """
    Abstract interface for a pose recovery method.

    Implementations must override _recover_pose which takes two sets of
    feature geometries and returns a PoseRecoveryResult
    """

    # NOTE(nathan) documented with discussed API from earlier
    def recover_pose(
        self,
        query_camera: PinholeCamera,
        query: VlcImage,
        match_camera: PinholeCamera,
        match: VlcImage,
        query_to_match: np.ndarray,
    ):
        """
        Recover pose from two frames and correspondences.

        Args:
            query_camera: Camera corresponding to query image
            query: Image and local features for query frame
            match_camera: Camera corresponding to match image
            match: Image and local features for match frame
            query_to_match: Nx2 correspondences between local features
        """
        if query.keypoints is None or match.keypoints is None:
            logging.error("Keypoints required for pose recovery!")
            return None

        # TODO(nathan) undistortion

        query_geometry = FeatureGeometry.from_image(
            query_camera, query, query_to_match[:, 0]
        )
        match_geometry = FeatureGeometry.from_image(
            match_camera, match, query_to_match[:, 1]
        )
        return self._recover_pose(query_geometry, match_geometry)

    @abc.abstractmethod
    def _recover_pose(self, query: FeatureGeometry, match: FeatureGeometry):
        """
        Implement actual pose recovery from feature geometry.

        Args:
            query: Feature bearings and associated landmarks if available for query
            match: Feature bearings and associated landmarks if available for match

        Returns:
            PoseRecoveryResult: match_T_query and associated information
        """
        pass  # pragma: no cover
