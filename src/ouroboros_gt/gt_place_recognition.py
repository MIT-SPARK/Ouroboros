import numpy as np

import ouroboros as ob


def get_gt_place_model():
    return GtPlaceModel()


class GtPlaceModel:

    def __init__(self):

        self.embedding_size = 8
        self.lc_recent_lockout_ns = 20 * 1e9
        self.lc_distance_threshold = 5

    def similarity_metric(self, embedding_query, embedding_stored):
        query_pose = ob.VlcPose.from_descriptor(embedding_query)
        stored_pose = ob.VlcPose.from_descriptor(embedding_stored)

        if stored_pose.time_ns > query_pose.time_ns - self.lc_recent_pose_lockout_ns:
            return np.inf

        d = np.linalg.norm(query_pose.position - stored_pose.position)

        if d > self.lc_distance_threshold:
            return -np.inf
        else:
            return -d

    def infer(self, image: ob.SparkImage, pose_hint: ob.VlcPose):
        if pose_hint is None:
            raise Exception("GtPlaceModel requires setting pose_hint")
        return pose_hint.to_descriptor()
