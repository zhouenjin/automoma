"""Extract task articulation metadata directly from a RealAppliance USD.

The extractor deliberately consumes semantic annotations and authored physics
relationships instead of asset identifiers.  It therefore produces the same
contract for an unseen asset without adding a per-asset branch.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple

import numpy as np

from .akr_adapter import AkrAttachmentSpec
from .contracts import ArticulationTaskSpec, JointKind
from .g2_adapter import Hand
from .transform_math import (
    Matrix4,
    freeze_matrix,
    invert_rigid,
    matrix_to_transform_rpy,
    quaternion_transform,
    transform_point,
    transform_vector,
)


_TASK_KEYWORDS = {
    "open": ("door", "drawer", "lid"),
    "press": ("button", "switch"),
    "rotate": ("knob", "dial"),
}


@dataclass(frozen=True)
class ExtractedUsdTask:
    """A normalized task plus the frames required to construct an AKR chain."""

    task_name: str
    semantic_label: str
    annotated_part: str
    joint_path: str
    body0_path: str
    body1_path: str
    task: ArticulationTaskSpec
    joint_axis_local: Tuple[float, float, float]
    world_from_joint_at_initial: Matrix4
    world_from_target_link_at_initial: Matrix4
    world_from_object_root: Matrix4

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result["task"]["joint_kind"] = self.task.joint_kind.value
        return result

    def make_attachment_spec(self, hand: Hand, world_from_ee: Sequence[Sequence[float]]) -> AkrAttachmentSpec:
        """Create the grasp-conditioned inverse chain for one EE pose."""

        world_from_ee_array = np.asarray(world_from_ee, dtype=np.float64)
        world_from_target = np.asarray(self.world_from_target_link_at_initial, dtype=np.float64)
        world_from_joint = np.asarray(self.world_from_joint_at_initial, dtype=np.float64)
        world_from_root = np.asarray(self.world_from_object_root, dtype=np.float64)
        return AkrAttachmentSpec(
            hand=hand,
            source_joint_name=self.task.joint_name,
            joint_kind=self.task.joint_kind,
            joint_axis=self.joint_axis_local,
            lower_limit=self.task.lower_limit,
            upper_limit=self.task.upper_limit,
            initial_position=self.task.initial_position,
            ee_to_handle=matrix_to_transform_rpy(invert_rigid(world_from_ee_array) @ world_from_target),
            handle_to_joint_at_initial=matrix_to_transform_rpy(invert_rigid(world_from_target) @ world_from_joint),
            joint_at_initial_to_object_root=matrix_to_transform_rpy(invert_rigid(world_from_joint) @ world_from_root),
        )

    def placed(self, world_from_source: Sequence[Sequence[float]]) -> "ExtractedUsdTask":
        """Place the source asset in a world frame without changing its articulation."""

        placement = np.asarray(world_from_source, dtype=np.float64)
        source_from_joint = np.asarray(self.world_from_joint_at_initial, dtype=np.float64)
        source_from_target = np.asarray(self.world_from_target_link_at_initial, dtype=np.float64)
        source_from_root = np.asarray(self.world_from_object_root, dtype=np.float64)
        task = ArticulationTaskSpec(
            joint_name=self.task.joint_name,
            target_link=self.task.target_link,
            handle_link=self.task.handle_link,
            joint_kind=self.task.joint_kind,
            axis=transform_vector(placement, self.task.axis),
            pivot=transform_point(placement, self.task.pivot),
            lower_limit=self.task.lower_limit,
            upper_limit=self.task.upper_limit,
            initial_position=self.task.initial_position,
        )
        return ExtractedUsdTask(
            task_name=self.task_name,
            semantic_label=self.semantic_label,
            annotated_part=self.annotated_part,
            joint_path=self.joint_path,
            body0_path=self.body0_path,
            body1_path=self.body1_path,
            task=task,
            joint_axis_local=self.joint_axis_local,
            world_from_joint_at_initial=freeze_matrix(placement @ source_from_joint),
            world_from_target_link_at_initial=freeze_matrix(placement @ source_from_target),
            world_from_object_root=freeze_matrix(placement @ source_from_root),
        )


def resolve_annotated_parts(annotations: Mapping[str, str], task_name: str) -> Tuple[Tuple[str, str], ...]:
    """Return every semantically compatible part, preserving annotation order."""

    task_key = task_name.strip().lower()
    if task_key not in _TASK_KEYWORDS:
        raise ValueError(f"unsupported task {task_name!r}; expected one of {sorted(_TASK_KEYWORDS)}")
    keywords = _TASK_KEYWORDS[task_key]
    matches = []
    seen = set()
    for label, part in annotations.items():
        normalized = label.strip().lower()
        if any(keyword in normalized for keyword in keywords) and part not in seen:
            matches.append((label, part))
            seen.add(part)
    if not matches:
        raise ValueError(f"no {task_key!r} part found in semantic annotations")
    return tuple(matches)


def _single_relationship_target(prim: Any, name: str) -> str:
    targets = prim.GetRelationship(name).GetTargets()
    if len(targets) != 1:
        raise ValueError(f"{prim.GetPath()}: {name} must have exactly one target, got {len(targets)}")
    return str(targets[0])


def _attribute_value(prim: Any, name: str) -> Any:
    attribute = prim.GetAttribute(name)
    value = attribute.Get() if attribute else None
    if value is None:
        raise ValueError(f"{prim.GetPath()}: required attribute {name!r} is missing")
    return value


def _quaternion_wxyz(quaternion: Any) -> Tuple[float, float, float, float]:
    imaginary = quaternion.GetImaginary()
    return float(quaternion.GetReal()), float(imaginary[0]), float(imaginary[1]), float(imaginary[2])


def _usd_world_matrix(xform_cache: Any, prim: Any) -> np.ndarray:
    # Gf uses row vectors; transpose into this package's column-vector convention.
    return np.asarray(xform_cache.GetLocalToWorldTransform(prim), dtype=np.float64).T


def _joint_kind(prim: Any) -> JointKind:
    type_name = prim.GetTypeName()
    if type_name == "PhysicsRevoluteJoint":
        return JointKind.REVOLUTE
    if type_name == "PhysicsPrismaticJoint":
        return JointKind.PRISMATIC
    raise ValueError(f"{prim.GetPath()}: unsupported joint type {type_name!r}")


def _joint_limits(prim: Any, kind: JointKind) -> Tuple[float, float]:
    lower = float(_attribute_value(prim, "physics:lowerLimit"))
    upper = float(_attribute_value(prim, "physics:upperLimit"))
    if kind is JointKind.REVOLUTE:
        return math.radians(lower), math.radians(upper)
    return lower, upper


def _initial_position(prim: Any, kind: JointKind) -> float:
    state_name = "state:angular:physics:position" if kind is JointKind.REVOLUTE else "state:linear:physics:position"
    state = prim.GetAttribute(state_name)
    value = state.Get() if state else None
    if value is None:
        return 0.0
    return math.radians(float(value)) if kind is JointKind.REVOLUTE else float(value)


def _iter_target_joints(stage: Any, target_body_path: str) -> Iterable[Any]:
    for prim in stage.Traverse():
        if prim.GetTypeName() not in ("PhysicsRevoluteJoint", "PhysicsPrismaticJoint"):
            continue
        targets = [str(path) for path in prim.GetRelationship("physics:body1").GetTargets()]
        if targets == [target_body_path]:
            yield prim


def _extract_joint(stage: Any, xform_cache: Any, task_name: str, label: str, part: str) -> ExtractedUsdTask:
    target_body_path = f"/World/{part}"
    joints = tuple(_iter_target_joints(stage, target_body_path))
    if len(joints) != 1:
        raise ValueError(f"{target_body_path}: expected one authored target joint, found {len(joints)}")
    joint = joints[0]
    kind = _joint_kind(joint)
    body0_path = _single_relationship_target(joint, "physics:body0")
    body1_path = _single_relationship_target(joint, "physics:body1")
    body0 = stage.GetPrimAtPath(body0_path)
    body1 = stage.GetPrimAtPath(body1_path)
    if not body0 or not body1:
        raise ValueError(f"{joint.GetPath()}: a joint body target does not exist")

    world_from_body0 = _usd_world_matrix(xform_cache, body0)
    world_from_body1 = _usd_world_matrix(xform_cache, body1)
    local_pos0 = tuple(float(value) for value in _attribute_value(joint, "physics:localPos0"))
    local_pos1 = tuple(float(value) for value in _attribute_value(joint, "physics:localPos1"))
    body0_from_joint = quaternion_transform(local_pos0, _quaternion_wxyz(_attribute_value(joint, "physics:localRot0")))
    body1_from_joint = quaternion_transform(local_pos1, _quaternion_wxyz(_attribute_value(joint, "physics:localRot1")))
    world_from_joint0 = world_from_body0 @ body0_from_joint
    world_from_joint1 = world_from_body1 @ body1_from_joint
    if not np.allclose(world_from_joint0, world_from_joint1, atol=2e-5):
        difference = float(np.max(np.abs(world_from_joint0 - world_from_joint1)))
        raise ValueError(f"{joint.GetPath()}: inconsistent authored joint frames (max error {difference:.3g})")

    axis_token = str(_attribute_value(joint, "physics:axis")).upper()
    local_axes = {"X": (1.0, 0.0, 0.0), "Y": (0.0, 1.0, 0.0), "Z": (0.0, 0.0, 1.0)}
    if axis_token not in local_axes:
        raise ValueError(f"{joint.GetPath()}: invalid joint axis {axis_token!r}")
    local_axis = local_axes[axis_token]
    world_axis = transform_vector(world_from_joint0, local_axis)
    world_pivot = transform_point(world_from_joint0, (0.0, 0.0, 0.0))
    lower, upper = _joint_limits(joint, kind)
    initial = _initial_position(joint, kind)
    task = ArticulationTaskSpec(
        joint_name=str(joint.GetPath()),
        target_link=body1_path,
        handle_link=body1_path,
        joint_kind=kind,
        axis=world_axis,
        pivot=world_pivot,
        lower_limit=lower,
        upper_limit=upper,
        initial_position=initial,
    )
    return ExtractedUsdTask(
        task_name=task_name,
        semantic_label=label,
        annotated_part=part,
        joint_path=str(joint.GetPath()),
        body0_path=body0_path,
        body1_path=body1_path,
        task=task,
        joint_axis_local=local_axis,
        world_from_joint_at_initial=freeze_matrix(world_from_joint0),
        world_from_target_link_at_initial=freeze_matrix(world_from_body1),
        world_from_object_root=freeze_matrix(world_from_body0),
    )


def extract_usd_tasks(usd_path: Path, annotation_path: Path, task_name: str = "open") -> Tuple[ExtractedUsdTask, ...]:
    """Extract all task-compatible articulations from one RealAppliance asset."""

    try:
        from pxr import Usd, UsdGeom
    except ImportError as error:  # pragma: no cover - exercised in the remote compatibility environment
        raise RuntimeError("USD extraction requires the pxr Python package") from error

    annotations = json.loads(Path(annotation_path).read_text(encoding="utf-8"))
    valid_annotations = isinstance(annotations, dict) and all(
        isinstance(key, str) and isinstance(value, str) for key, value in annotations.items()
    )
    if not valid_annotations:
        raise ValueError("gt_part.json must contain a string-to-string object")
    stage = Usd.Stage.Open(str(usd_path))
    if stage is None:
        raise ValueError(f"could not open USD stage {usd_path}")
    xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    return tuple(
        _extract_joint(stage, xform_cache, task_name, label, part)
        for label, part in resolve_annotated_parts(annotations, task_name)
    )
