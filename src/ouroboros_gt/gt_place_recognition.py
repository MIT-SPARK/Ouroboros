from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import ouroboros as ob
from ouroboros.config import Config, register_config


class GtPlaceModel:
    def __init__(self, config: GtPlaceModelConfig):
        self.embedding_size = config.embedding_size
        self.lc_recent_pose_lockout_ns = config.lc_recent_lockout_s * 1e9
        self.lc_distance_threshold = config.lc_distance_threshold

    def similarity_metric(self, embedding_query, embedding_stored):
        query_pose = ob.VlcPose.from_descriptor(embedding_query)
        stored_pose = ob.VlcPose.from_descriptor(embedding_stored)

        if stored_pose.time_ns > query_pose.time_ns - self.lc_recent_pose_lockout_ns:
            return -np.inf

        d = np.linalg.norm(query_pose.position - stored_pose.position)

        if d > self.lc_distance_threshold:
            return -np.inf
        else:
            return -d

    def infer(self, image: ob.SparkImage, pose_hint: ob.VlcPose):
        if pose_hint is None:
            raise Exception("GtPlaceModel requires setting pose_hint")
        return pose_hint.to_descriptor()


@register_config("place_model", name="ground_truth", constructor=GtPlaceModel)
@dataclass
class GtPlaceModelConfig(Config):
    embedding_size: int = 8
    lc_recent_lockout_s: float = 20
    lc_distance_threshold: float = 5
