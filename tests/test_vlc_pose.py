"""Test pose utilities."""

import numpy as np
import pytest

import ouroboros as ob


def test_pose_from_quat():
    """Test that pose creation from quaternion is sane."""
    T_identity = ob.pose_from_quat_trans(np.array([0, 0, 0, 1.0]), np.zeros((3, 1)))
    assert T_identity == pytest.approx(np.eye(4))

    T_non_identity = ob.pose_from_quat_trans(
        np.array([1, 0, 0, 0]), np.array([1, 2, 3])
    )
    expected = np.array([[1, 0, 0, 1], [0, -1, 0, 2], [0, 0, -1, 3], [0, 0, 0, 1]])
    assert T_non_identity == pytest.approx(expected)


def test_pose_inverse():
    """Test that pose inversion works."""
    a_T_b = ob.pose_from_quat_trans(np.array([1, 0, 0, 0]), np.array([1, 2, 3]))
    b_T_a = ob.invert_pose(a_T_b)
    assert a_T_b @ b_T_a == pytest.approx(np.eye(4))


def test_pose_descriptors():
    """Test that pose conversion is sane-ish via roundtrip conversion."""
    desc = np.array([5.0, 1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0])
    pose = ob.VlcPose.from_descriptor(desc)
    assert desc == pytest.approx(pose.to_descriptor())
