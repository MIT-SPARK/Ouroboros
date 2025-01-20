from dataclasses import dataclass
import numpy as np
from scipy.spatial.transform import Rotation as R


def invert_pose(p):
    p_inv = np.zeros((4, 4))
    p_inv[:3, :3] = p[:3, :3].T
    p_inv[:3, 3] = -p[:3, :3].T @ p[:3, 3]
    p_inv[3, 3] = 1
    return p_inv


def pose_from_quat_trans(q, t):
    pose = np.zeros((4, 4))
    Rmat = R.from_quat(q).as_matrix()
    pose[:3, :3] = Rmat
    pose[:3, 3] = t
    pose[3, 3] = 1
    return pose


@dataclass
class VlcPose:
    time_ns: int
    position: np.ndarray  # x,y,z
    rotation: np.ndarray  # qx, qy, qz

    def to_descriptor(self):
        return np.hstack([[self.time_ns], self.position, self.rotation])

    @classmethod
    def from_descriptor(cls, d):
        return cls(time_ns=d[0], position=d[1:4], rotation=d[4:])
