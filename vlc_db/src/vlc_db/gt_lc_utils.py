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


def compute_descriptor_similarity(
    lc_recent_pose_lockout_ns, lc_distance_threshold, d_query, d_stored
):
    query_pose = VlcPose.from_descriptor(d_query)
    stored_pose = VlcPose.from_descriptor(d_stored)

    if stored_pose.time_ns > query_pose.time_ns - lc_recent_pose_lockout_ns:
        return np.inf

    d = np.linalg.norm(query_pose.position - stored_pose.position)

    if d > lc_distance_threshold:
        return -np.inf
    else:
        return -d


def recover_pose(query_descriptors, match_descriptors):
    query_pose = VlcPose.from_descriptor(query_descriptors[0])
    match_pose = VlcPose.from_descriptor(match_descriptors[0])

    w_T_cur = pose_from_quat_trans(query_pose.rotation, query_pose.position)
    w_T_old = pose_from_quat_trans(match_pose.rotation, match_pose.position)

    old_T_new = invert_pose(invert_pose(w_T_old) @ w_T_cur)

    return old_T_new
