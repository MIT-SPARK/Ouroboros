from dataclasses import dataclass

import numpy as np

from ouroboros.vlc_db.spark_image import SparkImage
from ouroboros.vlc_db.vlc_pose import VlcPose


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
    pose_hint: VlcPose = None

    def get_feature_depths(self):
        """
        Get depth corresponding to the keypoints for image features.

        Note that this has hard-to-detect artifacts from features at the boundary
        of an image. We clip all keypoints to be inside the image with the assumption
        that whatever is consuming the depths is robust to small misalignments.

        Args:
            data: Image to extract depth from

        Returns:
            Optiona[np.ndarray]: Depths for keypoints if possible to extract
        """
        if self.keypoints is None:
            return None

        if self.image.depth is None:
            return None

        # NOTE(nathan) this is ugly, but:
        #   - To index into the image we need to swap from (u, v) to (row, col)
        #   - Numpy frustratingly doesn't have a buffered get, so we can't zero
        #     out-of-bounds elements. This only gets used assuming an
        #     outlier-robust method, so it should be fine
        dims = self.image.depth.shape
        limit = (dims[1] - 1, dims[0] - 1)
        coords = np.clip(np.round(self.keypoints), a_min=[0, 0], a_max=limit).astype(
            np.int64
        )
        return self.image.depth[coords[:, 1], coords[:, 0]]
