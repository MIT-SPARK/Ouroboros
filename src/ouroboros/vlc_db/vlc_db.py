from datetime import datetime
from typing import Union
import numpy as np

from ouroboros.vlc_db.spark_image import SparkImage
from ouroboros.vlc_db.vlc_image import VlcImage

from ouroboros.vlc_db.vlc_image_table import VlcImageTable
from ouroboros.vlc_db.vlc_lc_table import LcTable
from ouroboros.vlc_db.vlc_session_table import SessionTable
from ouroboros.vlc_db.spark_loop_closure import SparkLoopClosure


class KeypointSizeException:
    pass


class VlcDb:
    def __init__(self, image_embedding_dimension):
        self._image_table = VlcImageTable(image_embedding_dimension)
        self._lc_table = LcTable()
        self._session_table = SessionTable()

    def add_image(
        self,
        session_id: str,
        image_timestamp: Union[int, datetime],
        image: SparkImage,
    ) -> str:
        return self._image_table.add_image(session_id, image_timestamp, image)

    def get_image(self, image_uuid: str) -> VlcImage:
        img = self._image_table.get_image(image_uuid)
        if img is None:
            raise KeyError("Image not in database")
        return img

    def get_image_keys(self) -> [str]:
        return self._image_table.get_image_keys()

    def iterate_images(self):
        for image in self._image_table.iterate_images():
            yield image

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

        return self._image_table.query_embeddings(embeddings, k, similarity_metric)

    def update_embedding(self, image_uuid: str, embedding):
        self._image_table.update_embedding(image_uuid, embedding)

    def update_keypoints(self, image_uuid: str, keypoints, descriptors=None):
        if descriptors is not None:
            if len(keypoints) != len(descriptors):
                raise KeypointSizeException()
        self._image_table.update_keypoints(
            image_uuid, keypoints, descriptors=descriptors
        )

    def get_keypoints(self, image_uuid: str):
        return self._image_table.get_keypoints(image_uuid)

    def drop_image(self, image_uuid: str):
        """This functionality is for marginalization / sparsification of history"""
        self._image_table.drop_image(image_uuid)

    def add_session(self, robot_id: int, sensor_id: int = 0, name: str = None) -> str:
        return self._session_table.add_session(robot_id, sensor_id, name)

    def insert_session(
        self,
        session_uuid: str,
        start_time: Union[int, datetime],
        robot_id: int,
        sensor_id: int = 0,
        name: str = None,
    ):
        return self._session_table.insert_session(
            session_uuid, start_time, robot_id, sensor_id, name
        )

    def get_session(self, session_uuid: str):
        return self._session_table.get_session(session_uuid)

    def add_lc(self, loop_closure: SparkLoopClosure, session_uuid, creation_time=None):
        return self._lc_table.add_lc(loop_closure, session_uuid, creation_time)

    def get_lc(self, lc_uuid: str):
        return self._lc_table.get_lc(lc_uuid)

    def iterate_lcs(self):
        for lc in self._lc_table.iterate_lcs():
            yield lc
