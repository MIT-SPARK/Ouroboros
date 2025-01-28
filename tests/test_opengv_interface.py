"""Test opengv solver."""

import numpy as np
import pytest

import ouroboros_opengv as ogv


def test_inverse_camera_matrix():
    """Check that explicit inverse is correct."""
    orig = np.array([[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]])
    result = ogv.inverse_camera_matrix(orig)
    assert result == pytest.approx(np.linalg.inv(orig))


def test_bearings():
    """Check that bearing math is correct."""
    K = np.array([[10.0, 0.0, 5.0], [0.0, 5.0, 2.5], [0.0, 0.0, 1.0]])
    features = np.array([[5.0, 2.5], [15.0, 2.5], [5.0, -2.5]])
    bearings = ogv.get_bearings(K, features)
    expected = np.array(
        [
            [0.0, 0.0, 1.0],
            [1.0 / np.sqrt(2), 0.0, 1.0 / np.sqrt(2)],
            [0.0, -1.0 / np.sqrt(2), 1.0 / np.sqrt(2)],
        ]
    )
    with np.printoptions(suppress=True):
        print(f"expect:\n{expected}")
        print(f"bearings:\n{bearings}")

    assert bearings == pytest.approx(expected)


def _shuffle_features(features):
    indices = np.arange(features.shape[0])
    np.random.shuffle(indices)
    return indices, features[indices, :].copy()


def test_solver():
    """Test that two-view geometry is called correct."""
    query_features = np.random.normal(size=(100, 2))
    query_bearings = ogv.get_bearings(np.eye(3), query_features)

    yaw = np.pi / 4.0
    match_R_query = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(yaw), -np.sin(yaw)],
            [0.0, np.sin(yaw), np.cos(yaw)],
        ]
    )
    match_t_query = np.array([1.0, -1.2, 0.8]).reshape((3, 1))
    match_bearings = query_bearings @ match_R_query.T + match_t_query.T
    match_features = match_bearings[:, :2] / match_bearings[:, 2, np.newaxis]

    indices = np.arange(query_bearings.shape[0])
    new_indices, match_features = _shuffle_features(match_features)

    # needs to be query -> match (so need indices that were used by shuffle for query)
    correspondences = np.vstack((new_indices, indices)).T

    match_T_query = ogv.recover_pose_opengv(
        np.eye(3),
        query_features,
        np.eye(3),
        match_features,
        correspondences,
        solver=ogv.Solver2d2d.NISTER,
    )

    t_expected = np.squeeze(match_t_query / np.linalg.norm(match_t_query))
    t_result = np.squeeze(match_T_query[:3, 3] / np.linalg.norm(match_T_query[:3, 3]))
    assert match_T_query[:3, :3] == pytest.approx(match_R_query)
    assert t_result == pytest.approx(t_expected)
