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
