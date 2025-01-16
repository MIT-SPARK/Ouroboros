import time
import uuid

from vlc_db.spark_loop_closure import SparkLoopClosure, SparkLoopClosureMetadata


class LcTable:
    def __init__(self):
        self._lc_store = {}

    def add_lc(self, loop_closure: SparkLoopClosure, session_uuid, creation_time=None):
        lc_uuid = str(uuid.uuid4())
        if creation_time is None:
            creation_time = int(time.time() * 1e9)
        metadata = SparkLoopClosureMetadata(
            lc_uuid=lc_uuid, session_uuid=session_uuid, creation_time=creation_time
        )
        loop_closure.metadata = metadata

        self._lc_store[lc_uuid] = loop_closure
        return lc_uuid

    def get_lc(self, lc_uuid: str) -> SparkLoopClosure:
        return self._lc_store[lc_uuid]

    def iterate_lcs(self):
        """Iterate through loop closures according to ascending computed timestamp"""
        ts_keys = [
            (lc.metadata.creation_time, key) for key, lc in self._lc_store.items()
        ]
        for _, uid in sorted(ts_keys):
            yield self.get_lc(uid)
