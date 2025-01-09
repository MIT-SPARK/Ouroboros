from datetime import datetime
import uuid
import numpy as np

from vlc_db.spark_image import SparkImage
from vlc_db.vlc_image import VlcImage, VlcImageMetadata
from vlc_db.invertible_vector_store import InvertibleVectorStore


class VlcImageTable:
    def __init__(self, image_embedding_dimension):
        self.metadata_store = {}
        self.image_store = {}
        self.embedding_store = InvertibleVectorStore(image_embedding_dimension)
        self.keypoints_store = {}
        self.descriptors_store = {}

    def get_image(self, image_uuid):
        if image_uuid not in self.metadata_store:
            raise Exception(f"Image {image_uuid} not found in VLC DB!")
        metadata = self.metadata_store[image_uuid]
        image = self.image_store[image_uuid]
        embedding = self.embedding_store[image_uuid]
        keypoints = self.keypoints_store[image_uuid]
        descriptors = self.descriptors_store[image_uuid]

        vlc_image = VlcImage(metadata, image, embedding, keypoints, descriptors)
        return vlc_image

    def query_embeddings(
        self, embeddings: np.ndarray, k: int
    ) -> ([[VlcImage]], [[float]]):
        """Embeddings is a NxD numpy array, where N is the number of queries and D is the descriptor size
        Queries for the top k matches.

        Returns the top k closest matches and the match distances
        """

        uuid_lists, distance_lists = self.embedding_store.query(embeddings, k)
        for uuids, distances in zip(uuid_lists, distance_lists):
            images = [self.get_image(uid) for uid in uuids]

        return images, distance_lists

    def add_image(
        self,
        session_id: str,
        image_timestamp: datetime,
        image: SparkImage,
    ) -> str:
        new_uuid = str(uuid.uuid4())
        metadata = VlcImageMetadata(
            image_uuid=new_uuid,
            session_id=session_id,
            timestamp=image_timestamp,
        )
        self.metadata_store[new_uuid] = metadata
        self.image_store[new_uuid] = image
        return new_uuid

    def update_embedding(self, image_uuid: str, embedding):
        self.embedding_store[image_uuid] = embedding

    def update_keypoints(self, image_uuid: str, keypoints, descriptors=None):
        self.keypoints_store[image_uuid] = keypoints
        if descriptors is not None:
            assert len(keypoints) == len(descriptors)
        self.descriptors_store[image_uuid] = descriptors

    def drop_image(self, image_uuid: str):
        """This functionality is for marginalization / sparsification of history"""
        raise NotImplementedError("Cannot yet drop images from ImageStore")
