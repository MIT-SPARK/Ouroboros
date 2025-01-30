"""Module containing interfaces for pose recovery."""

import abc
import logging

import numpy as np

from ouroboros.vlc_db.vlc_db import VlcDb
from ouroboros.vlc_db.vlc_image import VlcImage

Matrix3d = np.ndarray
# TODO(nathan) use actual type alias once we move beyond 3.8
# Matrix3d = np.ndarray[np.float64[3, 3]]


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
        Tuple[np.ndarray, Optional[np.ndarray]]: Bearing vectors (and optionally points) in a Nx3 matrix.
    """
    K_inv = inverse_camera_matrix(K)
    versors = np.hstack((features, np.ones((features.shape[0], 1))))
    versors = versors @ K_inv.T
    # for broadcasting to be correct needs to be [N, 1] to multiply/divide rowwise
    points = None if depths is None else versors * depths[..., np.newaxis]
    bearings = versors / np.linalg.norm(versors, axis=1)[..., np.newaxis]
    return bearings, points


def _get_feature_depths(data: VlcImage):
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


class PoseRecovery(abc.ABC):
    """
    Abstract interface for a pose recovery method.

    Implementations must provide...
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

        cam_query = vlc_db.get_camera(query.metadata)
        cam_match = vlc_db.get_match(match.metadata)

        # TODO(nathan) undistortion

        depths_q = _get_feature_depths(query)[query_to_match[:, 0]]
        keypoints_q = query.keypoints[query_to_match[:, 0], :]
        bearings_q, points_q = get_bearings(cam_query.K, keypoints_q, depths_q)

        depths_m = _get_feature_depths(query)[query_to_match[:, 1]]
        keypoints_m = match.keypoints[query_to_match[:, 1], :]
        bearings_m, points_m = get_bearings(cam_match.K, keypoints_m, depths_m)

        # TODO(nathan) bundle everything into tuples and dispatch to solver
        # (hard to do masking based on invalid depth on the points themselves)

        if points_q is None and points_m is None:
            return self.recover_nonmetric_pose(bearings_q, bearings_m)
        elif points_m is None:
            raise NotImplementedError("TODO(nathan) swap query and match")
        else:
            return self.recover_pose(bearings_q, bearings_m)
