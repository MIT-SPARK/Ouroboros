from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, List, Optional, Tuple

import ouroboros as ob
from ouroboros.config import Config, config_field, register_config
from ouroboros.utils.plotting_utils import display_image_pair, display_kp_match_pair


class VlcServer:
    def __init__(
        self,
        config: VlcServerConfig,
        robot_id=0,
    ):
        self.lc_frame_lockout_ns = config.lc_frame_lockout_s * 1e9
        self.place_match_threshold = config.place_match_threshold

        self.display_place_matches = config.display_place_matches
        self.display_keypoint_matches = config.display_keypoint_matches

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

        self.vlc_db = ob.VlcDb(self.place_model.embedding_size)
        self.session_id = self.vlc_db.add_session(robot_id)

    def add_and_query_frame(
        self, image: ob.SparkImage, time_ns: int, pose_hint: ob.VlcPose = None
    ) -> Tuple[str, Optional[List[ob.SparkLoopClosure]]]:
        embedding = self.place_model.infer(image, pose_hint)

        image_matches, similarities = self.vlc_db.query_embeddings_max_time(
            embedding,
            1,
            time_ns - self.lc_frame_lockout_ns,
            similarity_metric=self.place_model.similarity_metric,
        )

        img_id = self.vlc_db.add_image(
            self.session_id, time_ns, image, pose_hint=pose_hint
        )
        vlc_image = self.vlc_db.update_embedding(img_id, embedding)

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
                img_id, image_keypoints, image_descriptors
            )

        if len(similarities) == 0 or similarities[0] < self.place_match_threshold:
            image_match = None
        else:
            image_match = image_matches[0]

        if self.display_place_matches:
            if image_match is None:
                right = None
            else:
                right = image_match.image.rgb
            display_image_pair(vlc_image.image.rgb, right)

        # TODO: support multiple possible place descriptor matches
        if image_match is not None:
            if not self.strict_keypoint_evaluation:
                # Since we just added the current image, we know that no keypoints
                # or descriptors have been generated for it
                image_keypoints, image_descriptors = self.keypoint_model.infer(
                    vlc_image.image, pose_hint=pose_hint
                )
                if image_descriptors is None:
                    image_descriptors = self.descriptor_model.infer(
                        vlc_image.image, image_keypoints, pose_hint=pose_hint
                    )
                vlc_image = self.vlc_db.update_keypoints(
                    img_id, image_keypoints, image_descriptors
                )

            # The matched frame may not yet have any keypoints or descriptors.
            if image_match.keypoints is None:
                keypoints, descriptors = self.keypoint_model.infer(
                    image_match.image, pose_hint=image_match.pose_hint
                )
            else:
                keypoints = image_match.keypoints

            if image_match.descriptors is None:
                if descriptors is None:
                    descriptors = self.descriptor_model.infer(
                        image_match.image, keypoints, pose_hint=image_match.pose_hint
                    )
                image_match = self.vlc_db.update_keypoints(
                    image_match.metadata.image_uuid, keypoints, descriptors
                )

            # Match keypoints
            img_kp_matched, stored_img_kp_matched = self.match_model.infer(
                vlc_image, image_match
            )
            if self.display_keypoint_matches:
                display_kp_match_pair(
                    vlc_image, image_match, img_kp_matched, stored_img_kp_matched
                )

            # 3. extract pose
            # TODO: matched keypoints go into pose_estimate
            pose_estimate = self.pose_model.infer(vlc_image, image_match)
            lc = ob.SparkLoopClosure(
                from_image_uuid=img_id,
                to_image_uuid=image_match.metadata.image_uuid,
                f_T_t=pose_estimate,
                quality=1,
            )
            lc_uid = self.vlc_db.add_lc(
                lc, self.session_id, creation_time=datetime.now()
            )
            lc_list = [self.vlc_db.get_lc(lc_uid)]

        else:
            lc_list = None

        return img_id, lc_list

    def get_lc_times(self, lc_uuid: str) -> Tuple[int, int]:
        lc = self.vlc_db.get_lc(lc_uuid)
        from_time_ns = self.vlc_db.get_image(lc.from_image_uuid).metadata.epoch_ns
        to_time_ns = self.vlc_db.get_image(lc.to_image_uuid).metadata.epoch_ns
        return from_time_ns, to_time_ns


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
    display_place_matches: bool = True
    display_keypoint_matches: bool = True

    @classmethod
    def load(cls, path: str):
        return ob.config.Config.load(VlcServerConfig, path)
