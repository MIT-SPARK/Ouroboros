import uuid
from datetime import datetime
from typing import Union

import numpy as np

from ouroboros.vlc_db.invertible_vector_store import InvertibleVectorStore
from ouroboros.vlc_db.spark_image import SparkImage
from ouroboros.vlc_db.utils import epoch_ns_from_datetime
from ouroboros.vlc_db.vlc_image import VlcImage, VlcImageMetadata
from ouroboros.vlc_db.vlc_pose import VlcPose


class VlcImageTable:
    def __init__(self, image_embedding_dimension):
        self.metadata_store = {}
        self.image_store = {}
        self.embedding_store = InvertibleVectorStore(image_embedding_dimension)
        self.keypoints_store = {}
        self.descriptors_store = {}
        self.pose_store = {}

    def get_image(self, image_uuid):
        if image_uuid not in self.metadata_store:
            return None
        metadata = self.metadata_store[image_uuid]
        image = self.image_store[image_uuid]
        if image_uuid in self.embedding_store:
            embedding = self.embedding_store[image_uuid]
        else:
            embedding = None
        keypoints = self.keypoints_store[image_uuid]
        descriptors = self.descriptors_store[image_uuid]
        pose = self.pose_store[image_uuid]

        vlc_image = VlcImage(metadata, image, embedding, keypoints, descriptors, pose)
        return vlc_image

    def get_image_keys(self):
        ts_keys = [
            (metadata.epoch_ns, key) for key, metadata in self.metadata_store.items()
        ]
        return [key for _, key in sorted(ts_keys)]

    def iterate_images(self):
        """Iterate through images according to ascending timestamp"""
        for key in self.get_image_keys():
            yield self.get_image(key)

    def query_embeddings(
        self,
        embeddings: np.ndarray,
        k: int,
        similarity_metric: Union[str, callable] = "ip",
    ) -> ([[VlcImage]], [[float]]):
        """Embeddings is a NxD numpy array, where N is the number of queries and D is the descriptor size
        Queries for the top k matches.

        Returns the top k closest matches and the match distances
        """

        uuid_lists, distance_lists = self.embedding_store.query(
            embeddings, k, similarity_metric
        )
        images = []
        for uuids, distances in zip(uuid_lists, distance_lists):
            images.append([self.get_image(uid) for uid in uuids])

        return images, distance_lists

    def add_image(
        self,
        session_id: str,
        image_timestamp: Union[int, datetime],
        image: SparkImage,
        pose_hint: VlcPose = None,
    ) -> str:
        new_uuid = str(uuid.uuid4())

        if isinstance(image_timestamp, datetime):
            image_timestamp = epoch_ns_from_datetime(image_timestamp)

        metadata = VlcImageMetadata(
            image_uuid=new_uuid,
            session_id=session_id,
            epoch_ns=image_timestamp,
        )
        self.metadata_store[new_uuid] = metadata
        self.image_store[new_uuid] = image
        self.pose_store[new_uuid] = pose_hint
        self.keypoints_store[new_uuid] = None
        self.descriptors_store[new_uuid] = None
        return new_uuid

    def update_embedding(self, image_uuid: str, embedding):
        self.embedding_store[image_uuid] = embedding

    def update_keypoints(self, image_uuid: str, keypoints, descriptors=None):
        self.keypoints_store[image_uuid] = keypoints
        if descriptors is not None:
            assert len(keypoints) == len(descriptors)
        self.descriptors_store[image_uuid] = descriptors

    def get_keypoints(self, image_uuid: str):
        return self.keypoints_store[image_uuid], self.descriptors_store[image_uuid]

    def drop_image(self, image_uuid: str):
        """This functionality is for marginalization / sparsification of history"""
        raise NotImplementedError("Cannot yet drop images from ImageStore")
