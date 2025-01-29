"""Module containing interfaces for pose recovery."""

import numpy as np

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


def get_bearings(K: Matrix3d, features: np.ndarray) -> np.ndarray:
    """
    Get bearings for undistorted features in pixel space.

    Args:
        K: Camera matrix for features.
        features: Pixel coordinates in a Nx2 matrix.

    Returns:
        Bearing vectors in a Nx3 matrix.
    """
    K_inv = inverse_camera_matrix(K)
    bearings = np.hstack((features, np.ones((features.shape[0], 1))))
    bearings = bearings @ K_inv.T
    # for broadcasting to be correct needs to be [N, 1] to divide rowwise
    bearings /= np.linalg.norm(bearings, axis=1)[..., np.newaxis]
    return bearings


def get_points(K: Matrix3d, features: np.ndarray, depths) -> np.ndarray:
    """
    Get bearings for undistorted features in pixel space.

    Args:
        K: Camera matrix for features.
        features: Pixel coordinates in a Nx2 matrix.

    Returns:
        Bearing vectors in a Nx3 matrix.
    """
    K_inv = inverse_camera_matrix(K)
    versors = np.hstack((features, np.ones((features.shape[0], 1))))
    versors = versors @ K_inv.T
    return versors * depths[..., np.newaxis]
