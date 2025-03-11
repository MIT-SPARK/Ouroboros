from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any, List, Optional, Tuple, Union

import numpy as np
from spark_config import Config, config_field, register_config

import ouroboros as ob
from ouroboros.vlc_db.utils import epoch_ns_from_datetime


class VlcServer:
    def __init__(
        self,
        config: VlcServerConfig,
        robot_id=0,
        log_path=None,
    ):
        self.robot_id = 0
        self.lc_frame_lockout_ns = config.lc_frame_lockout_s * 1e9
        self.place_match_threshold = config.place_match_threshold

        self.strict_keypoint_evaluation = config.strict_keypoint_evaluation

        self.place_model = config.place_method.create()
        self.keypoint_model = config.keypoint_method.create()
        self.descriptor_model = config.descriptor_method.create()
        if self.descriptor_model is None:
            print(
                "Desciptor method set to None. Hopefully your keypoint detector returns descriptors too..."
            )

        self.match_model = config.match_method.create()
        self.pose_model = config.pose_method.create()
        self.display_method = config.display_method.create()
        if self.display_method:
            self.display_method.setup(log_path)

        self.vlc_db = ob.VlcDb(self.place_model.embedding_size)

    def register_camera(
        self,
        sensor_id: int,
        calibration: ob.PinholeCamera,
        calibration_time: Union[datetime, int],
        name: str = None,
    ) -> str:
        session_id = self.vlc_db.add_session(self.robot_id, sensor_id)
        if isinstance(calibration_time, datetime):
            calibration_time = epoch_ns_from_datetime(calibration_time)
        self.vlc_db.add_camera(session_id, calibration, calibration_time)
        return session_id

    def add_frame(
        self,
        session_id: str,
        image: ob.SparkImage,
        time_ns: int,
        pose_hint: ob.VlcPose = None,
    ) -> str:
        # Compute embedding
        embedding = self.place_model.infer(image, pose_hint)

        # Add image and embedding
        image_id = self.vlc_db.add_image(
            session_id, time_ns, image, pose_hint=pose_hint
        )
        vlc_image = self.vlc_db.update_embedding(image_id, embedding)

        if self.strict_keypoint_evaluation:
            # Optionally force all keypoints/descriptors to be computed when
            # frame is added to db, and not lazily upon finding match
            image_keypoints, image_descriptors = self.keypoint_model.infer(
                vlc_image.image, pose_hint
            )
            if image_descriptors is None:
                image_descriptors = self.descriptor_model.infer(
                    vlc_image.image, image_keypoints, pose_hint
                )
            vlc_image = self.vlc_db.update_keypoints(
                image_id, image_keypoints, image_descriptors
            )
        return image_id

    def add_embedding_no_image(
        self,
        session_id: str,
        embedding: np.ndarray,
        time_ns: int,
        pose_hint: ob.VlcPose = None,
    ) -> str:
        # Add image and embedding
        image_id = self.vlc_db.add_image(session_id, time_ns, None, pose_hint=pose_hint)
        self.vlc_db.update_embedding(image_id, embedding)
        return image_id

    def find_match(
        self, image_id: str, time_ns: int, search_sessions: Optional[List[str]] = None
    ) -> Optional[str]:
        vlc_image = self.vlc_db.get_image(image_id)
        if vlc_image.embedding is None:
            # TODO(Yun) print warning
            return None

        # Find matches and similarity
        image_matches, similarities = self.vlc_db.query_embeddings_max_time(
            vlc_image.embedding,
            1,
            [vlc_image.metadata.session_id],
            time_ns - self.lc_frame_lockout_ns,
            similarity_metric=self.place_model.similarity_metric,
            search_sessions=search_sessions,
        )

        if len(similarities) == 0 or similarities[0] < self.place_match_threshold:
            image_match = None
        else:
            image_match = image_matches[0]

        if self.display_method:
            self.display_method.display_image_pair(vlc_image, image_match, time_ns)
        return image_match

    def compute_keypoints_descriptors(
        self, image_id: str, compute_depths=False
    ) -> ob.VlcImage:
        vlc_image = self.vlc_db.get_image(image_id)
        updated = False
        # The matched frame may not yet have any keypoints or descriptors.
        if vlc_image.keypoints is None:
            keypoints, descriptors = self.keypoint_model.infer(
                vlc_image.image, pose_hint=vlc_image.pose_hint
            )
            updated = True
        else:
            keypoints = vlc_image.keypoints

        if vlc_image.descriptors is None:
            if descriptors is None:
                descriptors = self.descriptor_model.infer(
                    vlc_image.image, keypoints, pose_hint=vlc_image.pose_hint
                )
                updated = True
        if updated:
            vlc_image = self.vlc_db.update_keypoints(image_id, keypoints, descriptors)
            if compute_depths:
                keypoint_depths = vlc_image.get_feature_depths()
                vlc_image = self.vlc_db.update_keypoint_depths(
                    image_id, keypoint_depths
                )
        return vlc_image

    def update_keypoints_decriptors(
        self, image_id: str, keypoints: np.ndarray, descriptors: np.ndarray
    ):
        self.vlc_db.update_keypoints(image_id, keypoints, descriptors)

    def update_keypoint_depths(self, image_id: str, keypoint_depths: np.ndarray):
        self.vlc_db.update_keypoint_depths(image_id, keypoint_depths)

    def compute_loop_closure_pose(
        self,
        session_id: str,
        query_image_id: str,
        match_image_id: str,
        time_ns: int,
    ) -> Optional[List[ob.SparkLoopClosure]]:
        query_image = self.vlc_db.get_image(query_image_id)
        match_image = self.vlc_db.get_image(match_image_id)
        # Match keypoints
        img_kp_matched, stored_img_kp_matched, query_to_match = self.match_model.infer(
            query_image, match_image
        )
        if self.display_method:
            self.display_method.display_kp_match_pair(
                query_image, match_image, img_kp_matched, stored_img_kp_matched, time_ns
            )

        # Extract pose
        query_camera = self.vlc_db.get_camera(query_image.metadata)
        match_camera = self.vlc_db.get_camera(match_image.metadata)
        pose_estimate = self.pose_model.recover_pose(
            query_camera.camera,
            query_image,
            match_camera.camera,
            match_image,
            query_to_match,
        )
        if not pose_estimate:
            return None

        if self.display_method:
            self.display_method.display_inlier_kp_match_pair(
                query_image,
                match_image,
                query_to_match,
                pose_estimate.inliers,
                time_ns,
            )
        lc = ob.SparkLoopClosure(
            from_image_uuid=query_image_id,
            to_image_uuid=match_image.metadata.image_uuid,
            f_T_t=pose_estimate.match_T_query,
            is_metric=pose_estimate.is_metric,
            quality=1,
        )
        lc_uid = self.vlc_db.add_lc(lc, session_id, creation_time=datetime.now())
        return [self.vlc_db.get_lc(lc_uid)]

    def add_and_query_frame(
        self,
        session_id: str,
        image: ob.SparkImage,
        time_ns: int,
        pose_hint: ob.VlcPose = None,
    ) -> Tuple[str, Optional[List[ob.SparkLoopClosure]]]:
        # Add to database and compute embedding (and optionally keypoints and descriptors)
        query_img_id = self.add_frame(session_id, image, time_ns, pose_hint)

        # Find match using the embeddings.
        image_match = self.find_match(query_img_id, time_ns)

        # TODO: support multiple possible place descriptor matches
        if image_match is None:
            return query_img_id, None

        self.compute_keypoints_descriptors(query_img_id)
        self.compute_keypoints_descriptors(image_match.metadata.image_uuid)

        lc_list = self.compute_loop_closure_pose(
            session_id, query_img_id, image_match.metadata.image_uuid, time_ns
        )

        return query_img_id, lc_list

    def get_lc_times(self, lc_uuid: str) -> Tuple[int, int]:
        lc = self.vlc_db.get_lc(lc_uuid)
        from_time_ns = self.vlc_db.get_image(lc.from_image_uuid).metadata.epoch_ns
        to_time_ns = self.vlc_db.get_image(lc.to_image_uuid).metadata.epoch_ns
        return from_time_ns, to_time_ns

    def has_image(self, image_uuid: str) -> bool:
        return self.vlc_db.has_image(image_uuid)

    def get_image(self, image_uuid: str) -> Optional[ob.VlcImage]:
        return self.vlc_db.get_image(image_uuid)


@register_config("vlc_server", name="vlc_server", constructor=VlcServer)
@dataclass
class VlcServerConfig(Config):
    place_method: Any = config_field("place_model", default="Salad")
    keypoint_method: Any = config_field("keypoint_model", default="SuperPoint")
    descriptor_method: Any = config_field("descriptor_model", required=False)
    match_method: Any = config_field("match_model", default="Lightglue")
    pose_method: Any = config_field("pose_model", default="ground_truth")
    lc_frame_lockout_s: int = 10
    place_match_threshold: float = 0.5
    strict_keypoint_evaluation: bool = False
    display_method: Any = config_field("vlc_server_display", required=False)

    @classmethod
    def load(cls, path: str):
        return Config.load(VlcServerConfig, path)


class VlcServerOpenCvDisplay:
    def __init__(self, config: VlcServerOpenCvDisplay):
        self.is_setup = False
        self.config = config
        if (
            config.save_place_matches
            or config.save_keypoint_matches
            or config.save_inlier_keypoint_matches
        ):
            if config.save_dir is None:
                raise Exception("You asked to save images but provided no save_dir")
            if not os.path.exists(config.save_dir):
                os.mkdir(config.save_dir)

    def setup(self, log_path=None):
        self.is_setup = True
        if log_path is not None:
            self.save_dir = log_path
        else:
            self.save_dir = os.path.join(
                self.config.save_dir, datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")
            )
            os.mkdir(self.save_dir)
        self.image_pair_dir = os.path.join(self.save_dir, "image_pairs")
        os.mkdir(self.image_pair_dir)
        self.separate_image_dir = os.path.join(self.save_dir, "separate_images")
        os.mkdir(self.separate_image_dir)
        self.kp_pair_dir = os.path.join(self.save_dir, "kp_matches")
        os.mkdir(self.kp_pair_dir)
        self.inlier_pair_dir = os.path.join(self.save_dir, "inlier_matches")
        os.mkdir(self.inlier_pair_dir)

    def display_image_pair(
        self, left: ob.SparkImage, right: Optional[ob.SparkImage], time_ns: int
    ):
        assert self.is_setup, "Must call `setup` first"
        if self.config.display_place_matches:
            ob.utils.plotting_utils.display_image_pair(left, right)

        if self.config.save_place_matches:
            if not self.config.only_save_positive_matches or right is not None:
                save_fn = os.path.join(
                    self.image_pair_dir, f"place_matches_{time_ns}.png"
                )
                ob.utils.plotting_utils.save_image_pair(left, right, save_fn)

        if self.config.save_separate_place_matches:
            if not self.config.only_save_positive_matches or right is not None:
                save_fn_l = os.path.join(self.separate_image_dir, f"{time_ns}_left.png")
                save_fn_r = os.path.join(
                    self.separate_image_dir, f"{time_ns}_right.png"
                )
                ob.utils.plotting_utils.save_image(save_fn_l, left)
                ob.utils.plotting_utils.save_image(save_fn_r, right)

    def display_kp_match_pair(
        self, left: ob.VlcImage, right: ob.VlcImage, left_kp, right_kp, time_ns: int
    ):
        assert self.is_setup, "Must call `setup` first"
        if self.config.display_keypoint_matches:
            ob.utils.plotting_utils.display_kp_match_pair(
                left, right, left_kp, right_kp
            )

        if self.config.save_keypoint_matches:
            save_fn = os.path.join(self.kp_pair_dir, f"kp_matches_{time_ns}.png")
            ob.utils.plotting_utils.save_kp_match_pair(
                left, right, left_kp, right_kp, save_fn
            )

    def display_inlier_kp_match_pair(
        self,
        left: ob.VlcImage,
        right: ob.VlcImage,
        query_to_match,
        inliers,
        time_ns: int,
    ):
        assert self.is_setup, "Must call `setup` first"
        inlier_mask = np.zeros(len(query_to_match), dtype=bool)
        inlier_mask[inliers] = True
        left_inliers = left.keypoints[query_to_match[inlier_mask, 0]]
        right_inliers = right.keypoints[query_to_match[inlier_mask, 1]]
        left_outliers = left.keypoints[query_to_match[np.logical_not(inlier_mask), 0]]
        right_outliers = right.keypoints[query_to_match[np.logical_not(inlier_mask), 1]]
        if self.config.display_inlier_keypoint_matches:
            ob.utils.plotting_utils.display_inlier_kp_match_pair(
                left, right, left_inliers, right_inliers, left_outliers, right_outliers
            )

        if self.config.save_inlier_keypoint_matches:
            save_fn = os.path.join(
                self.inlier_pair_dir, f"inlier_kp_matches_{time_ns}.png"
            )
            ob.utils.plotting_utils.save_inlier_kp_match_pair(
                left,
                right,
                left_inliers,
                right_inliers,
                left_outliers,
                right_outliers,
                save_fn,
            )


@register_config(
    "vlc_server_display", name="opencv", constructor=VlcServerOpenCvDisplay
)
@dataclass
class VlcServerOpenCvDisplayConfig(Config):
    display_place_matches: bool = True
    save_place_matches: bool = False
    save_separate_place_matches: bool = False
    only_save_positive_matches: bool = True

    display_keypoint_matches: bool = True
    save_keypoint_matches: bool = False

    display_inlier_keypoint_matches: bool = True
    save_inlier_keypoint_matches: bool = False

    save_dir: Optional[str] = None
