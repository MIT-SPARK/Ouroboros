from typing import Optional

import numpy as np

from _ouroboros_opengv import Solver2d2d, Solver2d3d, solve_2d2d, solve_2d3d, solve_3d3d

# TODO(nathan) use actual type alias once we move beyond 3.8
# Matrix3d = np.ndarray[np.float64[3, 3]]
Matrix3d = np.ndarray


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
        features: Pixel coordinates in a 2xN matrix.

    Returns:
        Bearing vectors in a 3xN matrix.
    """
    K_inv = inverse_camera_matrix(K)
    bearings = np.vstack((features, np.ones(features.shape[1])))
    bearings = K_inv @ bearings
    bearings /= np.linalg.norm(bearings, axis=0)
    return bearings


def recover_pose_opengv(
    K_query: Matrix3d,
    query_features: np.ndarray,
    K_match: Matrix3d,
    match_features: np.ndarray,
    correspondences: np.ndarray,
    solver=Solver2d2d.STEWENIUS,
) -> Optional[np.ndarray]:
    """
    Recover pose up to scale from 2d correspondences.

    Args:
        K_query: Camera matrix for query features.
        query_features: 2xN matrix of pixel features for query frame.
        K_match: Camera matrix for match features.
        match_features: 2xN matrix of pixel feature for match frame.
        correspondences: Nx2 indices of feature matches (query -> match)
        solver: Underlying 2d2d algorithm.

    Returns:
        match_T_query if underlying solver is successful.
    """
    query_bearings = get_bearings(K_query, query_features[:, correspondences[:, 0]])
    match_bearings = get_bearings(K_match, match_features[:, correspondences[:, 1]])
    # order is src (query), dest (match) for dest_T_src (match_T_query)
    result = solve_2d2d(query_bearings, match_bearings, solver=solver)
    if not result:
        return None

    match_T_query = result.dest_T_src
    return match_T_query
