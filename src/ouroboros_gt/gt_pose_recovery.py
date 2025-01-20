from ouroboros.vlc_db.vlc_pose import pose_from_quat_trans, invert_pose, VlcPose
import ouroboros as ob


def recover_pose(query_descriptors, match_descriptors):
    query_pose = VlcPose.from_descriptor(query_descriptors[0])
    match_pose = VlcPose.from_descriptor(match_descriptors[0])

    w_T_cur = pose_from_quat_trans(query_pose.rotation, query_pose.position)
    w_T_old = pose_from_quat_trans(match_pose.rotation, match_pose.position)

    old_T_new = invert_pose(invert_pose(w_T_old) @ w_T_cur)

    return old_T_new


# import numpy as np


def get_gt_pose_model():
    return GtPoseModel()


class GtPoseModel:

    def __init__(self):
        pass

    def infer(self, query_image: ob.VlcImage, match_image: ob.VlcImage):
        return recover_pose(query_image.descriptors, match_image.descriptors)
