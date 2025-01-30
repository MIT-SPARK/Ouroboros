from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from _ouroboros_opengv import (
    Solver2d2d,
    Solver2d3d,
    recover_translation_2d3d,
    solve_2d2d,
    solve_2d3d,
    solve_3d3d,
)
from ouroboros.config import Config
from ouroboros.pose_recovery import get_bearings, get_points

Matrix3d = np.ndarray


@dataclass
class RansacConfig(Config):
    """
    RANSAC parameters used when recovering pose.

    Attributes:
        max_iterations: Maximum number of RANSAC iterations to perform
        inlier_tolerance: Inlier reprojection tolerance for model-fitting
        inlier_probability: Probability of drawing at least one inlier during model selection
        min_inliers: Minimum number of inliers
    """

    max_iterations: int = 1000
    inlier_tolerance: float = 1.0e-2
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
        use_pnp_for_scale: Toggles between P2P and Arun's method for scale recovery
        scale_ransac: RANSAC parameters for translation recovery
        min_cosine_similarity: Minimum similarity threshold to translation w/o scale
    """

    solver: str = "STEWENIUS"
    ransac: RansacConfig = field(default_factory=RansacConfig)
    scale_recovery: bool = True
    use_pnp_for_scale: bool = True
    scale_ransac: RansacConfig = field(default_factory=RansacConfig)
    min_cosine_similarity: float = 0.8


class OpenGVPoseRecovery:
    """Class for performing pose recovery."""

    def __init__(self, config: OpenGVPoseRecoveryConfig):
        """Initialize the opengv pose recovery class via a config."""
        self._config = config

    def recover_pose(
        self,
        K_query: Matrix3d,
        query_features: np.ndarray,
        K_match: Matrix3d,
        match_features: np.ndarray,
        correspondences: np.ndarray,
    ) -> Optional[np.ndarray]:
        """
        Recover pose up to scale from 2d correspondences.

        Args:
            K_query: Camera matrix for query features.
            query_features: 2xN matrix of pixel features for query frame.
            K_match: Camera matrix for match features.
            match_features: 2xN matrix of pixel feature for match frame.
            correspondences: Nx2 indices of feature matches (query -> match)
            solver: Underlying 2d2d algorithm.

        Returns:
            match_T_query if underlying solver is successful.
        """
        query_bearings = get_bearings(K_query, query_features[correspondences[:, 0], :])
        match_bearings = get_bearings(K_match, match_features[correspondences[:, 1], :])
        # order is src (query), dest (match) for dest_T_src (match_T_query)
        result = solve_2d2d(
            query_bearings.T, match_bearings.T, solver=self._config.solver
        )
        if not result:
            return None

        match_T_query = result.dest_T_src
        return match_T_query

    def recover_metric_pose(
        self,
        K_query: Matrix3d,
        query_features: np.ndarray,
        K_match: Matrix3d,
        match_features: np.ndarray,
        match_depths: np.ndarray,
        correspondences: np.ndarray,
    ) -> Optional[np.ndarray]:
        """
        Recover pose up to scale from 2d correspondences.

        Args:
            K_query: Camera matrix for query features.
            query_features: Nx2 matrix of pixel features for query frame.
            K_match: Camera matrix for match features.
            match_features: Nx2 matrix of pixel feature for match frame.
            match_depths: N depths for each pixel
            correspondences: Nx2 indices of feature matches (query -> match)
            solver: Underlying 2d2d algorithm.

        Returns:
            match_T_query if underlying solver is successful.
        """
        query_bearings = get_bearings(K_query, query_features[correspondences[:, 0], :])
        match_bearings = get_bearings(K_match, match_features[correspondences[:, 1], :])
        # order is src (query), dest (match) for dest_T_src (match_T_query)
        result = solve_2d2d(
            query_bearings.T, match_bearings.T, solver=self._config.solver
        )

        if not result:
            return None

        # TODO(nathan) handle masking invalid depth
        dest_R_src = result.dest_T_src[:3, :3]
        match_points = get_points(
            K_match,
            match_features[correspondences[:, 1], :],
            match_depths[correspondences[:, 1]],
        )

        result = recover_translation_2d3d(query_bearings.T, match_points.T, dest_R_src)
        if not result:
            return None  # TODO(nathan) handle failure with state enum

        return result.dest_T_src
