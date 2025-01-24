from dataclasses import dataclass
import numpy as np


@dataclass
class SparkLoopClosureMetadata:
    lc_uuid: str
    session_uuid: str
    creation_time: int


@dataclass
class SparkLoopClosure:
    from_image_uuid: str
    to_image_uuid: str
    f_T_t: np.ndarray  # 4x4
    quality: float
    metadata: SparkLoopClosureMetadata = None
