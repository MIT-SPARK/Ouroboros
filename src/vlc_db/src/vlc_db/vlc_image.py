from dataclasses import dataclass

import numpy as np

from vlc_db.spark_image import SparkImage


@dataclass
class VlcImageMetadata:
    image_uuid: str
    session_id: str
    epoch_ns: int


@dataclass
class VlcImage:
    metadata: VlcImageMetadata
    image: SparkImage
    embedding: np.ndarray = None
    keypoints: np.ndarray = None
    descriptors: np.ndarray = None
