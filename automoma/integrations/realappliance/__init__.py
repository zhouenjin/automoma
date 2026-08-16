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
    "PlanarBaseLimits",
    "TransformRPY",
    "build_g2_akr_urdf",
    "build_planar_g2_urdf",
    "fit_bounds_with_spheres",
    "extract_usd_tasks",
    "infer_link_visual_bounds",
    "generate_interaction_hypotheses",
    "load_contact_candidates",
    "make_g2_akr_config",
    "make_g2_curobo_config",
    "resolve_annotated_parts",
]
