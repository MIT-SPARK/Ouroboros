from dataclasses import dataclass
import numpy as np
import datetime


@dataclass
class SparkLoopClosureMetadata:
    lc_uuid: str
    session_uuid: str
    creation_time: datetime


@dataclass
class SparkLoopClosure:
    from_robot_id: int
    to_robot_id: int
    f_T_t: np.ndarray  # 4x4
    timestamp: datetime
    quality: float
    metadata: SparkLoopClosureMetadata = None
