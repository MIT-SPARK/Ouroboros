import uuid
from datetime import datetime
from typing import Union

from ouroboros.vlc_db.spark_session import SparkSession
from ouroboros.vlc_db.utils import epoch_ns_from_datetime


class SessionTable:
    def __init__(self):
        self._session_store = {}

    def add_session(self, robot_id: int, sensor_id: int = 0, name: str = None) -> str:
        # Used when the VLC module is in charge of handling session   logic

        session_uuid = str(uuid.uuid4())
        self.insert_session(session_uuid, datetime.now(), robot_id, sensor_id, name)
        return session_uuid

    def insert_session(
        self,
        session_uuid: str,
        start_time: Union[int, datetime],
        robot_id: int,
        sensor_id: int = 0,
        name: str = None,
    ) -> str:
        # Used when some external "universal state manager" handles session logic

        if name is None:
            name = session_uuid

        if isinstance(start_time, datetime):
            start_time = epoch_ns_from_datetime(start_time)

        spark_session = SparkSession(
            session_uuid=session_uuid,
            start_time=start_time,
            name=name,
            robot_id=robot_id,
            sensor_id=sensor_id,
        )
        self._session_store[session_uuid] = spark_session

    def get_session(self, session_uuid: str) -> SparkSession:
        return self._session_store[session_uuid]
