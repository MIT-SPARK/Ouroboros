from datetime import datetime
from typing import Callable, List, Optional, Tuple, TypeVar, Union

import numpy as np

from ouroboros.vlc_db.spark_image import SparkImage
from ouroboros.vlc_db.spark_loop_closure import SparkLoopClosure
from ouroboros.vlc_db.utils import epoch_ns_from_datetime
from ouroboros.vlc_db.vlc_image import VlcImage
from ouroboros.vlc_db.vlc_image_table import VlcImageTable
from ouroboros.vlc_db.vlc_lc_table import LcTable
from ouroboros.vlc_db.vlc_pose import VlcPose
from ouroboros.vlc_db.vlc_session_table import SessionTable


class KeypointSizeException(BaseException):
    pass


T = TypeVar("T")


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
        pose_hint: VlcPose = None,
    ) -> str:
        """image_timestamp should be a datetime object or integer number of nanoseconds"""
        return self._image_table.add_image(
            session_id, image_timestamp, image, pose_hint
        )

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
        embedding: np.ndarray,
        k: int,
        similarity_metric: Union[str, callable] = "ip",
    ) -> ([VlcImage], [float]):
        """Embeddings is a NxD numpy array, where N is the number of queries and D is the descriptor size
        Queries for the top k matches.

        Returns the top k closest matches and the match distances
        """

        assert embedding.ndim == 1, "Query embedding must be 1d vector"

        matches, similarities = self.batch_query_embeddings(
            np.array([embedding]), k, similarity_metric
        )
        return matches[0], similarities[0]

    def query_embeddings_filter(
        self,
        embedding: np.ndarray,
        k: int,
        filter_function: Callable[[VlcImage, float], bool],
        similarity_metric: Union[str, callable] = "ip",
    ):
        ret = self.batch_query_embeddings_filter(
            np.array([embedding]),
            k,
            lambda _, img, sim: filter_function(img, sim),
            similarity_metric,
        )

        return_matches = [(t[0], t[2]) for t in ret[0]]
        return return_matches

    def query_embeddings_max_time(
        self,
        embedding: np.ndarray,
        k: int,
        max_time: Union[float, int, datetime],
        similarity_metric: Union[str, callable] = "ip",
    ) -> ([VlcImage], [float]):
        """Query image embeddings to find the k closest vectors with timestamp older than max_time."""

        if isinstance(max_time, datetime):
            max_time = epoch_ns_from_datetime(max_time)
        # NOTE: This is a placeholder implementation. Ideally, we re-implement
        # this to be more efficient and not iterate through the full set of
        # vectors

        def time_filter(_, vlc_image, similarity):
            return vlc_image.metadata.epoch_ns < max_time

        ret = self.batch_query_embeddings_filter(np.array([embedding]), k, time_filter)
        matches = [t[2] for t in ret[0]]
        similarities = [t[0] for t in ret[0]]
        return matches, similarities

    def batch_query_embeddings_uuid_filter(
        self,
        uuids: [str],
        k: int,
        filter_function: Callable[[VlcImage, VlcImage, float], bool],
        similarity_metric: Union[str, callable] = "ip",
    ):
        # get image for each uuid and call query_embeddings)filter

        embeddings = np.array([self.get_image(u).embedding for u in uuids])
        images = [self.get_image(u) for u in uuids]
        return self.batch_query_embeddings_filter(
            embeddings, k, filter_function, similarity_metric, filter_metadata=images
        )

    def batch_query_embeddings(
        self,
        embeddings: np.ndarray,
        k: int,
        similarity_metric: Union[str, callable] = "ip",
    ) -> ([[VlcImage]], [[float]]):
        """Embeddings is a NxD numpy array, where N is the number of queries and D is the descriptor size
        Queries for the top k matches.

        Returns the top k closest matches and the match distances
        """

        assert embeddings.ndim == 2, (
            f"Batch query requires an NxD array of query embeddings, not {embeddings.shape}"
        )

        return self._image_table.query_embeddings(embeddings, k, similarity_metric)

    def batch_query_embeddings_filter(
        self,
        embeddings: np.ndarray,
        k: int,
        filter_function: Callable[[T, VlcImage, float], bool],
        similarity_metric: Union[str, callable] = "ip",
        filter_metadata: Optional[Union[T, List[T]]] = None,
    ) -> [[Tuple[float, T, VlcImage]]]:
        """Query image embeddings to find the k closest vectors that satisfy
        the filter function. Note that this query may be much slower than
        `query_embeddings_with_max_time` because it requires iterating over all
        stored images.
        """

        if isinstance(filter_metadata, list):
            assert len(filter_metadata) == len(embeddings)
        elif filter_metadata is None:
            filter_metadata = [None] * len(embeddings)
        elif callable(filter_metadata):
            filter_metadata = [filter_metadata] * len(embeddings)
        else:
            raise Exception("filter function must be a list, callable, or None")

        matches, similarities = self.batch_query_embeddings(
            embeddings, -1, similarity_metric="ip"
        )

        filtered_matches_out = []
        for metadata, matches_for_query, similarities_for_query in zip(
            filter_metadata, matches, similarities
        ):
            filtered_matches_for_query = []
            n_matches = 0
            for match_image, similarity in zip(
                matches_for_query, similarities_for_query
            ):
                if filter_function(metadata, match_image, similarity):
                    filtered_matches_for_query.append(
                        (similarity, metadata, match_image)
                    )
                    n_matches += 1

                if n_matches >= k:
                    break

            filtered_matches_out.append(filtered_matches_for_query)

        return filtered_matches_out

    def update_embedding(self, image_uuid: str, embedding):
        self._image_table.update_embedding(image_uuid, embedding)
        return self.get_image(image_uuid)

    def update_keypoints(self, image_uuid: str, keypoints, descriptors=None):
        if descriptors is not None:
            if len(keypoints) != len(descriptors):
                raise KeypointSizeException()
        self._image_table.update_keypoints(
            image_uuid, keypoints, descriptors=descriptors
        )
        return self.get_image(image_uuid)

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
