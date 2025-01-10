from dataclasses import dataclass
import numpy as np


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


def compute_descriptor_distance(
    lc_recent_pose_lockout_ns, lc_distance_threshold, d_query, d_stored
):
    query_pose = VlcPose.from_descriptor(d_query)
    stored_pose = VlcPose.from_descriptor(d_stored)

    if stored_pose.time_ns > query_pose.time_ns - lc_recent_pose_lockout_ns:
        return np.inf

    d = np.linalg.norm(query_pose.position - stored_pose.position)

    if d > lc_distance_threshold:
        return np.inf
    else:
        return d
