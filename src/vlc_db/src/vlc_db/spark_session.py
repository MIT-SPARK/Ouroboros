from dataclasses import dataclass


@dataclass
class SparkSession:
    session_uuid: str
    start_time: int
    name: str
    robot_id: int
    sensor_id: int
