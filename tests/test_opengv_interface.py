"""Test opengv solver."""

import numpy as np
import pytest

import ouroboros as ob
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


def test_nonmetric_solver():
    """Test that two-view geometry is called correctly."""
    expected = _get_test_pose()  # match_T_query
    query_features = np.random.normal(size=(100, 2))
    query_bearings, _ = ob.get_bearings(np.eye(3), query_features)
    match_bearings = _transform_points(query_bearings, expected)

    result = ogv.solve_2d2d(
        query_bearings.T,
        match_bearings.T,
        solver=ogv.Solver2d2d.NISTER,
    )

    _check_pose_up_to_scale(expected, result.dest_T_src)


# def test_metric_solver():
# """Test that 2d-3d solver is called correctly."""
# expected = _get_test_pose()  # match_T_query

# match_features = np.random.normal(size=(30, 2))
# match_depths = np.random.uniform(1.5, 2.5, match_features.shape[0])
# match_points = ogv.get_points(np.eye(3), match_features, match_depths)

# query_points = _transform_points(match_points, _inverse_pose(expected))
# query_features = query_points[:, :2] / query_points[:, 2, np.newaxis]

# indices = np.arange(query_points.shape[0])
# new_indices, query_features = _shuffle_features(query_features)

# # needs to be query -> match (so need indices that were used by shuffle for match)
# correspondences = np.vstack((indices, new_indices)).T
# result = ogv.recover_metric_pose_opengv(
# np.eye(3),
# query_features,
# np.eye(3),
# match_features,
# match_depths,
# correspondences,
# solver=ogv.Solver2d2d.STEWENIUS,
# )
# assert result == pytest.approx(expected)
