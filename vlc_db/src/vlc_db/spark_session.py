from dataclasses import dataclass
from datetime import datetime


@dataclass
class SparkSession:
    session_uuid: str
    start_time: datetime
    name: str
    robot_id: int
    sensor_id: int
