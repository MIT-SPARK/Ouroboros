import uuid
from typing import Optional

from ouroboros.vlc_db.camera import PinholeCamera, VlcCamera
from ouroboros.vlc_db.vlc_image import VlcImageMetadata


class CameraTable:
    def __init__(self):
        self._camera_store = {}

    def add_camera(
        self,
        session_id: str,
        camera: PinholeCamera,
        calibration_time: Optional[int] = None,
    ) -> str:
        camera_uuid = str(uuid.uuid4())
        vlc_camera = VlcCamera(
            session_id=session_id, camera=camera, calibration_epoch_ns=calibration_time
        )
        self._camera_store[camera_uuid] = vlc_camera
        return camera_uuid

    def get_camera(self, image_metadata: VlcImageMetadata) -> Optional[VlcCamera]:
        # return camera calibration closest in time to the image timestamp
        image_session = image_metadata.session_id
        cameras_for_session = [
            (k, v)
            for k, v in self._camera_store.items()
            if v.session_id == image_session
        ]
        if not cameras_for_session:
            return None

        dts = [
            (abs(image_metadata.epoch_ns - v.calibration_epoch_ns), k)
            for k, v in cameras_for_session
        ]
        return self._camera_store[sorted(dts)[0][1]]
