"""G2 + RealAppliance integration for the external-first experiment."""

from .contracts import ArticulationTaskSpec, JointKind, PhysicsRunPolicy
from .g2_adapter import Hand, PlanarBaseLimits, build_planar_g2_urdf, make_g2_curobo_config
from .hypotheses import BasePoseSeed, InteractionHypothesis, generate_interaction_hypotheses

__all__ = [
    "ArticulationTaskSpec",
    "BasePoseSeed",
    "Hand",
    "InteractionHypothesis",
    "JointKind",
    "PhysicsRunPolicy",
    "PlanarBaseLimits",
    "build_planar_g2_urdf",
    "generate_interaction_hypotheses",
    "make_g2_curobo_config",
]
