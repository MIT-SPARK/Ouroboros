from datetime import datetime
import numpy as np

from vlc_db.spark_image import SparkImage
from vlc_db.vlc_image import VlcImage

from vlc_db.vlc_image_table import VlcImageTable
from vlc_db.vlc_lc_table import LcTable
from vlc_db.vlc_session_table import SessionTable
from vlc_db.spark_loop_closure import SparkLoopClosure


class VlcDb:
    def __init__(self, image_embedding_dimension):
        self._image_table = VlcImageTable(image_embedding_dimension)
        self._lc_table = LcTable()
        self._session_table = SessionTable()

    def add_image(
        self,
        session_id: str,
        image_timestamp: datetime,
        image: SparkImage,
    ) -> str:
        return self._image_table.add_image(session_id, image_timestamp, image)

    def get_image(self, image_uuid):
        return self._image_table.get_image(image_uuid)

    def query_embeddings(
        self, embeddings: np.ndarray, k: int
    ) -> ([[VlcImage]], [[float]]):
        """Embeddings is a NxD numpy array, where N is the number of queries and D is the descriptor size
        Queries for the top k matches.

        Returns the top k closest matches and the match distances
        """

        return self._image_table.query_embeddings(embeddings, k)

    def update_embedding(self, image_uuid: str, embedding):
        self._image_table.update_embedding(image_uuid, embedding)

    def update_keypoints(self, image_uuid: str, keypoints, descriptors=None):
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
        self, robot_id: int, sensor_id: int = 0, name: str = None
    ) -> str:
        return self._session_table.insert_session(robot_id, sensor_id, name)

    def get_session(self, session_uuid: str):
        return self._session_table.get_session(session_uuid)

    def add_lc(self, loop_closure: SparkLoopClosure, session_uuid, creation_time=None):
        return self._lc_table.add_lc(loop_closure, session_uuid, creation_time)

    def get_lc(self, lc_uuid: str):
        return self._lc_table.get_lc(lc_uuid)
