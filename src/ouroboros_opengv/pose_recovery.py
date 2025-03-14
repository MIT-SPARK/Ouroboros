"""Module containing opengv pose recovery implementation."""

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from spark_config import Config, ConfigFactory

from ouroboros.pose_recovery import FeatureGeometry, PoseRecovery, PoseRecoveryResult
from ouroboros.vlc_db.vlc_pose import invert_pose
from ouroboros_opengv._ouroboros_opengv import (
    RansacResult,
    Solver2d2d,
    recover_translation_2d3d,
    solve_2d2d,
    solve_3d3d,
)

Logger = logging.getLogger(__name__)
Matrix3d = np.ndarray


@dataclass
class RansacConfig(Config):
    """
    RANSAC parameters used when recovering pose.

    Attributes:
        max_iterations: Maximum number of RANSAC iterations to perform
        inlier_tolerance: Inlier reprojection tolerance for model-fitting
        inlier_probability: Probability of drawing at least one inlier in model indices
        min_inliers: Minimum number of inliers
    """

    max_iterations: int = 1000
    inlier_tolerance: float = 1.0e-4
    inlier_probability: float = 0.99
    min_inliers: int = 10


@dataclass
class OpenGVPoseRecoveryConfig(Config):
    """
    Config for pose recovery.

    Attributes:
        solver: 2D-2D initial solver to use [STEWENIUS, NISTER, SEVENPT, EIGHTPT]
        ransac: RANSAC parameters for 2D-2D solver
        scale_recovery: Attempt to recover translation scale if possible
        mask_outliers: Mask rejected correspondences from 2d pose recovery
        use_pnp_for_scale: Toggles between P2P and Arun's method for scale recovery
        scale_ransac: RANSAC parameters for translation recovery
        min_cosine_similarity: Minimum similarity threshold to translation w/o scale
    """

    solver: str = "NISTER"
    ransac: RansacConfig = field(default_factory=RansacConfig)
    scale_recovery: bool = True
    mask_outliers: bool = True
    use_pnp_for_scale: bool = True
    scale_ransac: RansacConfig = field(default_factory=RansacConfig)
    min_cosine_similarity: Optional[float] = None

    @property
    def solver_enum(self):
        """Get enum from string."""
        if self.solver == "STEWENIUS":
            return Solver2d2d.STEWENIUS
        elif self.solver == "NISTER":
            return Solver2d2d.NISTER
        elif self.solver == "SEVENPT":
            return Solver2d2d.SEVENPT
        elif self.solver == "EIGHTPT":
            return Solver2d2d.EIGHTPT
        else:
            Logger.warning(
                "Invalid solver type '{self.solver}', defaulting to STEWENIUS!"
            )
            return Solver2d2d.STEWENIUS


class OpenGVPoseRecovery(PoseRecovery):
    """Class for performing pose recovery."""

    def __init__(self, config: OpenGVPoseRecoveryConfig):
        """Initialize the opengv pose recovery class via a config."""
        self._config = config

    @property
    def config(self):
        """Get underlying config."""
        return self._config

    @classmethod
    def load(cls, path):
        """Load opengv pose recovery from file."""
        config = Config.load(OpenGVPoseRecoveryConfig, path)
        return cls(config)

    def _recover_translation_3d3d(
        self, query: FeatureGeometry, match: FeatureGeometry, result_2d2d: RansacResult
    ):
        if not query.is_metric or not match.is_metric:
            Logger.warning(
                "Both inputs must have depth to use Arun's method for translation!"
            )
            return None

        match_valid = (match.depths > 0.0) & np.isfinite(match.depths)
        query_valid = (query.depths > 0.0) & np.isfinite(query.depths)
        valid = match_valid & query_valid
        if self._config.mask_outliers:
            inlier_mask = np.zeros_like(valid)
            inlier_mask[result_2d2d.inliers] = True
            valid &= inlier_mask

        # TODO(nathan) adjust tolerance for 3d-3d solver?
        return solve_3d3d(
            query.points.T,
            match.points.T,
            max_iterations=self._config.scale_ransac.max_iterations,
            threshold=self._config.scale_ransac.inlier_tolerance,
            probability=self._config.scale_ransac.inlier_probability,
            min_inliers=self._config.scale_ransac.min_inliers,
        )

    def _recover_translation_2d3d(
        self,
        query: FeatureGeometry,
        match: FeatureGeometry,
        dest_R_src: Matrix3d,
        inliers=np.ndarray,
    ):
        valid = (match.depths > 0.0) & np.isfinite(match.depths)
        if self._config.mask_outliers:
            inlier_mask = np.zeros_like(valid)
            inlier_mask[inliers] = True
            valid &= inlier_mask

        bearings = query.bearings[valid]
        points = match.points[valid]
        return recover_translation_2d3d(
            bearings.T,
            points.T,
            dest_R_src,
            max_iterations=self._config.scale_ransac.max_iterations,
            threshold=self._config.scale_ransac.inlier_tolerance,
            probability=self._config.scale_ransac.inlier_probability,
            min_inliers=self._config.scale_ransac.min_inliers,
        )

    def _check_result(self, pose1, pose2):
        if self._config.min_cosine_similarity is None:
            return True

        t1 = pose1[:3, 3] / np.linalg.norm(pose1[:3, 3])
        t2 = pose2[:3, 3] / np.linalg.norm(pose2[:3, 3])
        return t1.dot(t2) < self._config.min_cosine_similarity

    def _recover_pose(
        self, query: FeatureGeometry, match: FeatureGeometry
    ) -> PoseRecoveryResult:
        """Recover pose up to scale from 2d correspondences."""
        # order is src (query), dest (match) for dest_T_src (match_T_query)
        result = solve_2d2d(
            query.bearings.T,
            match.bearings.T,
            solver=self._config.solver_enum,
            max_iterations=self._config.ransac.max_iterations,
            threshold=self._config.ransac.inlier_tolerance,
            probability=self._config.ransac.inlier_probability,
            min_inliers=self._config.ransac.min_inliers,
        )
        if not result:
            return PoseRecoveryResult()  # by default, invalid

        nonmetric = not query.is_metric and not match.is_metric
        if not self._config.scale_recovery or nonmetric:
            return PoseRecoveryResult.nonmetric(result.dest_T_src, result.inliers)

        if not self._config.use_pnp_for_scale:
            metric_result = self._recover_translation_3d3d(query, match, result)
            if metric_result is None or not metric_result:
                return PoseRecoveryResult.nonmetric(result.dest_T_src, result.inliers)

            if not self._check_result(result.dest_T_src, metric_result.dest_T_src):
                return PoseRecoveryResult.nonmetric(result.dest_T_src, result.inliers)

            # NOTE(nathan) we override the recovered 3d-3d translation with the
            # more accurate 2d-2d rotation
            dest_T_src = metric_result.dest_T_src.copy()
            dest_T_src[:3, :3] = result.dest_T_src[:3, :3]
            return PoseRecoveryResult.metric(dest_T_src, result.inliers)

        dest_R_src = result.dest_T_src[:3, :3]
        need_inverse = False
        if not match.is_metric:
            # we need to invert the problem to get bearing vs. point order correct
            need_inverse = True
            metric_result = self._recover_translation_2d3d(
                match, query, dest_R_src.T, result.inliers
            )
        else:
            metric_result = self._recover_translation_2d3d(
                query, match, dest_R_src, result.inliers
            )

        if not metric_result:
            return PoseRecoveryResult.nonmetric(result.dest_T_src, result.inliers)

        dest_T_src = metric_result.dest_T_src
        if need_inverse:
            dest_T_src = invert_pose(dest_T_src)

        if not self._check_result(result.dest_T_src, dest_T_src):
            return PoseRecoveryResult.nonmetric(result.dest_T_src, result.inliers)

        return PoseRecoveryResult.metric(dest_T_src, result.inliers)


ConfigFactory.register(
    OpenGVPoseRecoveryConfig,
    "pose_model",
    name="opengv",
    constructor=OpenGVPoseRecovery,
)
