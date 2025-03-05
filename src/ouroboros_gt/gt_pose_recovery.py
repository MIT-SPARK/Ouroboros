from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from spark_config import Config, register_config

import ouroboros as ob
from ouroboros.vlc_db.vlc_pose import invert_pose, pose_from_quat_trans


def _recover_pose(query_pose, match_pose):
    w_T_query = pose_from_quat_trans(query_pose.rotation, query_pose.position)
    w_T_match = pose_from_quat_trans(match_pose.rotation, match_pose.position)
    match_T_query = invert_pose(w_T_match) @ w_T_query
    return match_T_query


class GtPoseModel:
    def __init__(self, config: GtPoseModel):
        pass

    def recover_pose(
        self,
        query_camera: ob.PinholeCamera,
        query: ob.VlcImage,
        match_camera: ob.PinholeCamera,
        match: ob.VlcImage,
        query_to_match: np.ndarray,
    ):
        p1 = query.pose_hint
        p2 = match.pose_hint
        match_T_query = _recover_pose(p1, p2)
        return ob.PoseRecoveryResult.metric(match_T_query)


@register_config("pose_model", name="ground_truth", constructor=GtPoseModel)
@dataclass
class GtPoseModelConfig(Config):
    pass
