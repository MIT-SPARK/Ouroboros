"""Test opengv solver."""

import numpy as np
import pytest

import ouroboros_opengv as ogv


def _transform_points(bearings, dest_T_src):
    return bearings @ dest_T_src[:3, :3].T + dest_T_src[:3, 3, np.newaxis].T


def _inverse_pose(dest_T_src):
    src_T_dest = np.eye(4)
    src_T_dest[:3, :3] = dest_T_src[:3, :3].T
    src_T_dest[:3, 3] = -dest_T_src[:3, :3].T @ dest_T_src[:3, 3]
    return src_T_dest


def _pose(dest_R_src, dest_t_src):
    dest_T_src = np.eye(4)
    dest_T_src[:3, :3] = dest_R_src
    dest_T_src[:3, 3] = np.squeeze(dest_t_src)
    return dest_T_src


def _get_test_pose():
    yaw = np.pi / 4.0
    match_R_query = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(yaw), -np.sin(yaw)],
            [0.0, np.sin(yaw), np.cos(yaw)],
        ]
    )
    # NOTE(nathan) needs to be small to avoid chirality inversion
    match_t_query = np.array([0.3, -0.2, 0.1]).reshape((3, 1))
    return _pose(match_R_query, match_t_query)


def _shuffle_features(features):
    indices = np.arange(features.shape[0])
    np.random.shuffle(indices)
    return indices, features[indices, :].copy()


def _check_pose_up_to_scale(expected, result):
    t_expected = expected[:3, 3]
    t_expected = t_expected / np.linalg.norm(t_expected)
    t_result = result[:3, 3]
    t_result = t_result / np.linalg.norm(t_result)
    assert result[:3, :3] == pytest.approx(expected[:3, :3], abs=1.0e-3)
    assert t_result == pytest.approx(t_expected, abs=1.0e-3)


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
    assert bearings == pytest.approx(expected)


def test_solver():
    """Test that two-view geometry is called correct."""
    query_features = np.random.normal(size=(100, 2))
    query_bearings = ogv.get_bearings(np.eye(3), query_features)

    expected = _get_test_pose()  # match_T_query
    match_bearings = _transform_points(query_bearings, expected)
    match_features = match_bearings[:, :2] / match_bearings[:, 2, np.newaxis]

    indices = np.arange(query_bearings.shape[0])
    new_indices, match_features = _shuffle_features(match_features)

    # needs to be query -> match (so need indices that were used by shuffle for query)
    correspondences = np.vstack((new_indices, indices)).T

    result = ogv.recover_pose_opengv(
        np.eye(3),
        query_features,
        np.eye(3),
        match_features,
        correspondences,
        solver=ogv.Solver2d2d.NISTER,
    )

    _check_pose_up_to_scale(expected, result)


def test_points():
    """Test that point conversion match is correct."""
    K = np.array([[10.0, 0.0, 5.0], [0.0, 5.0, 2.5], [0.0, 0.0, 1.0]])
    features = np.array([[5.0, 2.5], [15.0, 2.5], [5.0, -2.5]])
    depths = np.array([0.9, 2.0, 3.0])
    points = ogv.get_points(K, features, depths)
    expected = np.array(
        [
            [0.0, 0.0, 0.9],
            [2.0, 0.0, 2.0],
            [0.0, -3.0, 3.0],
        ]
    )
    with np.printoptions(suppress=True):
        print(f"expected:\n{expected}")
        print(f"points:\n{points}")

    assert points == pytest.approx(expected)


def test_metric_solver():
    """Test that two-view geometry is called correct."""
    expected = _get_test_pose()  # match_T_query

    match_features = np.random.normal(size=(30, 2))
    match_depths = np.random.uniform(1.5, 2.5, match_features.shape[0])
    match_points = ogv.get_points(np.eye(3), match_features, match_depths)

    query_points = _transform_points(match_points, _inverse_pose(expected))
    query_features = query_points[:, :2] / query_points[:, 2, np.newaxis]

    indices = np.arange(query_points.shape[0])
    new_indices, query_features = _shuffle_features(query_features)

    # needs to be query -> match (so need indices that were used by shuffle for match)
    correspondences = np.vstack((indices, new_indices)).T
    result = ogv.recover_metric_pose_opengv(
        np.eye(3),
        query_features,
        np.eye(3),
        match_features,
        match_depths,
        correspondences,
        solver=ogv.Solver2d2d.STEWENIUS,
    )

    with np.printoptions(suppress=True):
        print(f"expected:\n{expected}")
        print(f"result:\n{result}")

    assert result == pytest.approx(expected)
