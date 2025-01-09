from dataclasses import dataclass
from datetime import datetime

import numpy as np

from vlc_db.spark_image import SparkImage


@dataclass
class VlcImageMetadata:
    image_uuid: str
    session_id: str
    session_frame_id: int
    timestamp: datetime


@dataclass
class VlcImage:
    metadata: VlcImageMetadata
    image: SparkImage
    embedding: np.ndarray = None
    keypoints: np.ndarray = None
    descriptors: np.ndarray = None
