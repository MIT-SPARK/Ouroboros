from __future__ import annotations
from dataclasses import dataclass

import ouroboros as ob
from ouroboros.vlc_db.vlc_pose import invert_pose, pose_from_quat_trans
from ouroboros.config import Config, register_config


def recover_pose(query_pose, match_pose):
    w_T_cur = pose_from_quat_trans(query_pose.rotation, query_pose.position)
    w_T_old = pose_from_quat_trans(match_pose.rotation, match_pose.position)

    old_T_new = invert_pose(invert_pose(w_T_old) @ w_T_cur)

    return old_T_new


class GtPoseModel:
    def __init__(self, config: GtPoseModel):
        pass

    def infer(self, query_image: ob.VlcImage, match_image: ob.VlcImage):
        p1 = query_image.pose_hint
        p2 = match_image.pose_hint
        return recover_pose(p1, p2)


@register_config("pose_model", name="ground_truth", constructor=GtPoseModel)
@dataclass
class GtPoseModelConfig(Config):
    pass
