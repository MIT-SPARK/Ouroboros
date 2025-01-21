# TODO: move this to ouroboros, not ouroboros_ros
from datetime import datetime
from typing import List, Optional, Tuple

import ouroboros as ob
from ouroboros.utils.plotting_utils import display_image_pair


class VlcServer:
    def __init__(
        self,
        place_method,
        keypoint_method,
        descriptor_method,
        pose_method,
        lc_frame_lockout_ns,
        place_match_threshold,
        robot_id=0,
        strict_keypoint_evaluation=False,
    ):
        self.lc_frame_lockout_ns = lc_frame_lockout_ns
        self.place_match_threshold = place_match_threshold
        self.display_place_matches = True
        self.strict_keypoint_evaluation = strict_keypoint_evaluation

        # To be replaced with "virtual configs"
        if place_method == "salad":
            from ouroboros_salad.salad_model import get_salad_model

            self.place_model = get_salad_model()
        elif place_method == "ground_truth":
            from ouroboros_gt.gt_place_recognition import get_gt_place_model

            self.place_model = get_gt_place_model()
            raise NotImplementedError(
                "ground truth place descriptors not yet supported (but will be)"
            )
        else:
            raise NotImplementedError(f"Place descriptor {place_method} not supported")

        if keypoint_method == "ground_truth":
            from ouroboros_gt.gt_keypoint_detection import get_gt_keypoint_model

            self.keypoint_model = get_gt_keypoint_model()
        else:
            raise NotImplementedError(
                f"keypoint extraction method {keypoint_method} not supported"
            )

        if descriptor_method == "ground_truth":
            from ouroboros_gt.gt_descriptors import get_gt_descriptor_model

            self.descriptor_model = get_gt_descriptor_model()
        else:
            raise NotImplementedError(
                f"descriptor extraction method {descriptor_method} not supported"
            )

        if pose_method == "ground_truth":
            from ouroboros_gt.gt_pose_recovery import get_gt_pose_model

            self.pose_model = get_gt_pose_model()
        else:
            raise NotImplementedError(
                f"pose extraction method {pose_method} not supported"
            )

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
            image_keypoints = self.keypoint_model.infer(vlc_image.image, pose_hint)
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
                image_keypoints = self.keypoint_model.infer(
                    vlc_image.image, pose_hint=pose_hint
                )
                image_descriptors = self.descriptor_model.infer(
                    vlc_image.image, image_keypoints, pose_hint=pose_hint
                )
                vlc_image = self.vlc_db.update_keypoints(
                    img_id, image_keypoints, image_descriptors
                )

            # The matched frame may not yet have any keypoints or descriptors.
            if image_match.keypoints is None:
                keypoints = self.keypoint_model.infer(
                    image_match.image, pose_hint=image_match.pose_hint
                )
            else:
                keypoints = image_match.keypoints

            if image_match.descriptors is None:
                descriptors = self.descriptor_model.infer(
                    image_match.image, keypoints, pose_hint=image_match.pose_hint
                )
                image_match = self.vlc_db.update_keypoints(
                    image_match.metadata.image_uuid, keypoints, descriptors
                )

            # 3. extract pose
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
