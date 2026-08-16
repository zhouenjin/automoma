"""G2 + RealAppliance integration for the external-first experiment."""

from .contracts import ArticulationTaskSpec, JointKind, PhysicsRunPolicy
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

__all__ = [
    "ArticulationTaskSpec",
    "BasePoseSeed",
    "Bounds3D",
    "Hand",
    "InteractionHypothesis",
    "JointKind",
    "PhysicsRunPolicy",
    "PlanarBaseLimits",
    "build_planar_g2_urdf",
    "fit_bounds_with_spheres",
    "infer_link_visual_bounds",
    "generate_interaction_hypotheses",
    "make_g2_curobo_config",
]
