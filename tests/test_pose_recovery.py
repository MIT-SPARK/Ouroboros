"""Test image geometry functions."""

import numpy as np
import pytest

import ouroboros as ob


def test_inverse_camera_matrix():
    """Check that explicit inverse is correct."""
    orig = np.array([[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]])
    result = ob.inverse_camera_matrix(orig)
    assert result == pytest.approx(np.linalg.inv(orig))


def test_bearings():
    """Check that bearing math is correct."""
    K = np.array([[10.0, 0.0, 5.0], [0.0, 5.0, 2.5], [0.0, 0.0, 1.0]])
    features = np.array([[5.0, 2.5], [15.0, 2.5], [5.0, -2.5]])
    bearings, _ = ob.get_bearings(K, features)
    expected = np.array(
        [
            [0.0, 0.0, 1.0],
            [1.0 / np.sqrt(2), 0.0, 1.0 / np.sqrt(2)],
            [0.0, -1.0 / np.sqrt(2), 1.0 / np.sqrt(2)],
        ]
    )
    assert bearings == pytest.approx(expected)


def test_points():
    """Test that point conversion match is correct."""
    K = np.array([[10.0, 0.0, 5.0], [0.0, 5.0, 2.5], [0.0, 0.0, 1.0]])
    features = np.array([[5.0, 2.5], [15.0, 2.5], [5.0, -2.5]])
    depths = np.array([0.9, 2.0, 3.0])
    _, points = ob.get_bearings(K, features, depths)
    expected = np.array(
        [
            [0.0, 0.0, 0.9],
            [2.0, 0.0, 2.0],
            [0.0, -3.0, 3.0],
        ]
    )
    assert points == pytest.approx(expected)


def test_depth_extraction():
    """Test that depth extraction for keypoints works."""
    img = ob.VlcImage(None, ob.SparkImage())
    # no keypoints -> no depths
    assert ob.get_feature_depths(img) is None

    img.keypoints = np.array([[2, 1], [3.1, 1.9], [-1, 10], [10, -1]])
    # no depth image -> no depths
    assert ob.get_feature_depths(img) is None

    img.image.depth = np.arange(24).reshape((4, 6))
    depths = ob.get_feature_depths(img)
    # should be indices (1, 2), (2, 3), (3, 0), (0, 5)
    assert depths == pytest.approx(np.array([8, 15, 18, 5]))


def test_camera():
    """Test that camera matrix is correct."""
    camera = ob.Camera(ob.CameraConfig(5.0, 10.0, 3.0, 4.0))
    expected = np.array([[5.0, 0.0, 3.0], [0.0, 10.0, 4.0], [0.0, 0.0, 1.0]])
    assert camera.K == pytest.approx(expected)


def test_feature_geometry():
    """Test feature geometry creation."""
    camera = ob.Camera(ob.CameraConfig(2.0, 2.0, 4.0, 3.0))
    img = ob.VlcImage(None, ob.SparkImage())
    img.keypoints = np.array([[4.0, 3.0], [6.0, 3.0], [4.0, 1.0]])
    geometry = ob.FeatureGeometry.from_image(camera, img)

    expected_bearings = np.array(
        [
            [0.0, 0.0, 1.0],
            [1.0 / np.sqrt(2), 0.0, 1.0 / np.sqrt(2)],
            [0.0, -1.0 / np.sqrt(2), 1.0 / np.sqrt(2)],
        ]
    )
    assert not geometry.is_metric
    assert geometry.bearings == pytest.approx(expected_bearings)

    img.image.depth = np.zeros((6, 8))
    img.image.depth[3, 4] = 0.9
    img.image.depth[3, 6] = 2.0
    img.image.depth[1, 4] = 3.0
    geometry = ob.FeatureGeometry.from_image(camera, img)

    expected_depths = np.array([0.9, 2.0, 3.0])
    expected_points = np.array(
        [
            [0.0, 0.0, 0.9],
            [2.0, 0.0, 2.0],
            [0.0, -3.0, 3.0],
        ]
    )
    assert geometry.is_metric
    assert geometry.bearings == pytest.approx(expected_bearings)
    assert geometry.depths == pytest.approx(expected_depths)
    assert geometry.points == pytest.approx(expected_points)

    geometry = ob.FeatureGeometry.from_image(camera, img, np.array([1]))

    assert geometry.is_metric
    assert geometry.bearings == pytest.approx(expected_bearings[np.newaxis, 1])
    assert geometry.depths == pytest.approx(expected_depths[np.newaxis, 1])
    assert geometry.points == pytest.approx(expected_points[np.newaxis, 1])


def test_pose_result_construction():
    """Test that constructor invariants work."""
    invalid = ob.PoseRecoveryResult()
    assert not invalid

    valid = ob.PoseRecoveryResult(np.eye(4))
    assert valid

    metric = ob.PoseRecoveryResult.metric(np.eye(4))
    assert metric
    assert metric.is_metric

    nonmetric = ob.PoseRecoveryResult.nonmetric(np.eye(4))
    assert nonmetric
    assert not nonmetric.is_metric
