"""Test opengv solver."""

import unittest.mock as mock

import numpy as np
import pytest

import ouroboros as ob
import ouroboros_opengv as ogv


def _get_test_pose():
    angle = np.pi / 5.0
    match_R_query = np.array([np.sin(angle / 2), 0.0, 0.0, np.cos(angle / 2)])
    match_t_query = np.array([0.4, -0.3, 0.1]).reshape((3, 1))
    return ob.pose_from_quat_trans(match_R_query, match_t_query)


def _transform_points(bearings, dest_T_src):
    return bearings @ dest_T_src[:3, :3].T + dest_T_src[:3, 3, np.newaxis].T


def test_nonmetric_solver():
    """Test that two-view geometry is called correctly."""
    rng = np.random.default_rng(0)

    expected = _get_test_pose()  # match_T_query
    query_features = rng.normal(size=(100, 2))
    query_bearings, _ = ob.get_bearings(np.eye(3), query_features)
    match_bearings = _transform_points(query_bearings, expected)

    result = ogv.solve_2d2d(
        query_bearings.T,
        match_bearings.T,
        solver=ogv.Solver2d2d.NISTER,
    ).dest_T_src

    t_expected = expected[:3, 3]
    t_expected = t_expected / np.linalg.norm(t_expected)
    t_result = result[:3, 3]
    t_result = t_result / np.linalg.norm(t_result)
    assert result[:3, :3] == pytest.approx(expected[:3, :3], abs=1.0e-3)
    assert t_result == pytest.approx(t_expected, abs=1.0e-3)


def test_pose_recovery_interface():
    """Test that pose recovery interface logic is correct."""
    rng = np.random.default_rng(0)

    match = ob.VlcImage(None, ob.SparkImage())
    match.keypoints = rng.normal(size=(30, 2))

    query = ob.VlcImage(None, ob.SparkImage())
    query.keypoints = rng.normal(size=(30, 2))

    indices = np.arange(query.keypoints.shape[0])
    correspondences = np.vstack((indices, indices)).T

    config = ogv.OpenGVPoseRecoveryConfig()
    pose_recovery = ogv.OpenGVPoseRecovery(config)
    vlc_db = ob.VlcDb(10)

    cam = ob.PinholeCamera()

    empty_image = ob.VlcImage(None, None)
    result = pose_recovery.recover_pose(
        vlc_db, empty_image, empty_image, correspondences
    )
    assert result is None

    # for coverage until camera lookup works
    with pytest.raises(NotImplementedError):
        result = pose_recovery.recover_pose(vlc_db, match, query, correspondences)

    with pytest.raises(NotImplementedError):
        result = pose_recovery.recover_pose(vlc_db, match, query, correspondences, cam)

    # for coverage to double-check non-metric solver works
    result = pose_recovery.recover_pose(vlc_db, match, query, correspondences, cam, cam)
    assert result
    assert not result.is_metric


def test_pose_recovery():
    """Test that pose recovery is correct."""
    rng = np.random.default_rng(0)
    expected = _get_test_pose()  # match_T_query

    match = ob.VlcImage(None, ob.SparkImage())
    match.keypoints = rng.normal(size=(100, 2))
    match_depths = rng.uniform(1.5, 2.5, match.keypoints.shape[0])
    _, match_points = ob.get_bearings(np.eye(3), match.keypoints, match_depths)

    query = ob.VlcImage(None, ob.SparkImage())
    query_points = _transform_points(match_points, ob.invert_pose(expected))
    query.keypoints = query_points[:, :2] / query_points[:, 2, np.newaxis]

    indices = np.arange(query_points.shape[0])
    new_indices = np.arange(query.keypoints.shape[0])
    # rng.shuffle(new_indices)
    query.keypoints = query.keypoints[new_indices, :].copy()
    query_depths = query_points[new_indices, 2]

    # needs to be query -> match (so need indices that were used by shuffle for match)
    correspondences = np.vstack((indices, new_indices)).T

    config = ogv.OpenGVPoseRecoveryConfig()
    pose_recovery = ogv.OpenGVPoseRecovery(config)
    vlc_db = ob.VlcDb(10)

    cam = ob.PinholeCamera()

    depths = [query_depths, match_depths]
    with mock.patch.object(ob.VlcImage, "get_feature_depths", side_effect=depths):
        result = pose_recovery.recover_pose(
            vlc_db, query, match, correspondences, cam, cam
        )
        assert result
        assert result.is_metric
        assert result.query_T_match == pytest.approx(expected, abs=1.0e-3)

    # make sure that inverse solve works
    depths = [query_depths, None]
    with mock.patch.object(ob.VlcImage, "get_feature_depths", side_effect=depths):
        result = pose_recovery.recover_pose(
            vlc_db, query, match, correspondences, cam, cam
        )
        assert result
        assert result.is_metric
        assert result.query_T_match == pytest.approx(expected, abs=1.0e-3)

    # check non-metric return behavior
    depths = [query_depths, np.zeros_like(query_depths)]
    with mock.patch.object(ob.VlcImage, "get_feature_depths", side_effect=depths):
        result = pose_recovery.recover_pose(
            vlc_db, query, match, correspondences, cam, cam
        )
        assert result
        assert not result.is_metric

    # check 3d-3d solver
    config = ogv.OpenGVPoseRecoveryConfig()
    config.use_pnp_for_scale = False
    pose_recovery = ogv.OpenGVPoseRecovery(config)

    depths = [query_depths, match_depths]
    with mock.patch.object(ob.VlcImage, "get_feature_depths", side_effect=depths):
        result = pose_recovery.recover_pose(
            vlc_db, query, match, correspondences, cam, cam
        )
        assert result
        assert result.is_metric
        assert result.query_T_match == pytest.approx(expected, abs=1.0e-3)

    # 3d-3d solver fails without both depths
    depths = [query_depths, None]
    with mock.patch.object(ob.VlcImage, "get_feature_depths", side_effect=depths):
        result = pose_recovery.recover_pose(
            vlc_db, query, match, correspondences, cam, cam
        )
        assert result
        assert not result.is_metric
