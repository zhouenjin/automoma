"""G2 + RealAppliance integration for the external-first experiment."""

from .akr_adapter import AkrAttachmentSpec, TransformRPY, build_g2_akr_urdf, make_g2_akr_config
from .contracts import ArticulationTaskSpec, JointKind, PhysicsRunPolicy
from .contact_candidates import ContactCandidate, PlacedContactCandidate, load_contact_candidates
from .g2_adapter import (
    Bounds3D,
    Hand,
    PlanarBaseLimits,
    build_planar_g2_urdf,
    fit_bounds_with_spheres,
    infer_link_visual_bounds,
    make_g2_curobo_config,
)
from .hypotheses import BasePoseSeed, InteractionHypothesis, generate_interaction_hypotheses
from .usd_task import ExtractedUsdTask, extract_usd_tasks, resolve_annotated_parts
from .usd_collision import PlacedCollisionWorld, fixed_body_cluster, load_placed_collision_world
from .trajectory_selection import weighted_cspace_path_scores
from .g2_runtime import planner_robot_joint_names, swerve_inverse_kinematics

__all__ = [
    "AkrAttachmentSpec",
    "ArticulationTaskSpec",
    "BasePoseSeed",
    "Bounds3D",
    "ContactCandidate",
    "ExtractedUsdTask",
    "Hand",
    "InteractionHypothesis",
    "JointKind",
    "PhysicsRunPolicy",
    "PlacedContactCandidate",
    "PlacedCollisionWorld",
    "PlanarBaseLimits",
    "TransformRPY",
    "build_g2_akr_urdf",
    "build_planar_g2_urdf",
    "fit_bounds_with_spheres",
    "fixed_body_cluster",
    "extract_usd_tasks",
    "infer_link_visual_bounds",
    "generate_interaction_hypotheses",
    "load_contact_candidates",
    "load_placed_collision_world",
    "make_g2_akr_config",
    "make_g2_curobo_config",
    "planner_robot_joint_names",
    "resolve_annotated_parts",
    "swerve_inverse_kinematics",
    "weighted_cspace_path_scores",
]
