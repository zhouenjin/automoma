# AutoMoMa-native RealAppliance branch

This branch is the external-first comparison. It starts from upstream AutoMoMa
and uses an AutoMoMa-compatible Franka execution stack. It does not import the
G2 implementation or its controllers.

Native runs now generate their component cloud and GraspGen candidates directly
from the selected RealAppliance USD component. No G2 hypothesis, trajectory, or
video is a permitted input to the native planner.

## Scope

The first gate is handled `open` only. The scene may be empty and the appliance
may be suspended. A robot embodiment is an implementation choice; physical
success and automatic transfer are the acceptance criteria.

The runtime contract is:

1. generate a handled contact pose from articulation and geometry metadata;
2. plan robot motion to pre-contact and contact;
3. close the gripper under physics;
4. move the end effector with the target joint while preserving the measured
   handle-to-gripper transform;
5. command robot DOFs only and let the appliance joint move through contact;
6. audit contact, penetration, attachment use, target-joint writes, and final
   articulation progress;
7. save three distant camera views and the full trace for every success and
   representative failure.

For revolute doors the controller aims for 80% of the joint range and strict
dataset acceptance remains 70%. For prismatic joints the corresponding values
are 90% and 80%.

## Non-negotiable checks

- `drive` execution only; object DOFs must not appear in executable actions.
- no fixed joint, surface-gripper attachment, or runtime weld to the appliance;
- no asset-ID branch, fixed world coordinate, or manually tuned per-asset
  offset in the search/controller;
- contact may be unilateral or bilateral, but some robot-target contact must
  be measured during functional progress;
- an episode exceeding the penetration threshold is a failure even if the
  target joint reaches the requested angle.

## Environment note (2026-08-16)

The public source repository excludes generated robot, object, and scene assets.
The public `automoma/automoma-assets` dataset currently contains no downloadable
payload. The 4090 host has Isaac Sim 4.5 and NVIDIA driver 535, while upstream
`main` declares Isaac Sim 5.1, Python 3.11, CUDA 12.8, and driver 580. Therefore
the first executable uses the installed Isaac Sim runtime, Isaac Sim's Franka
asset, and RealAppliance USDs. Asset resolution is isolated behind an adapter so
the missing upstream Summit USD can replace it later without changing success
semantics.

## Artifact isolation

- AutoMoMa-branch runs use an `automoma_native` root only.
- Compatibility-executor failures and native-planner results use different
  subdirectories.
- G2 videos, plans, logs, and success counts are excluded from this branch.
- A success directory is created only after the strict physical and provenance
  audits pass; partial transfers are labelled and quarantined.

## Implemented native planning path (2026-08-16)

The executable native path is now:

1. inspect the USD articulation and group fixed descendants under each open
   joint;
2. rank open components from geometry and joint metadata;
3. sample a fresh point cloud for the selected component, retaining per-rigid-
   body provenance;
4. call the upstream GraspGen service and retain candidate rank, confidence,
   and source rigid body;
5. construct a grasp-specific reversed AKR chain from the USD parent/child
   anchors and rotations;
6. inherit AutoMoMa's adjacent-payload collision topology while preserving
   all other robot and component collision checks;
7. sample the target articulation from closed to the requested fraction, solve
   collision-feasible IK at every sample, and select a continuous path through
   those IK layers;
8. pass the resulting robot-only path to the physical Isaac executor and audit
   contact, penetration, object-joint writes, and attachments.

The waypoint search is a generic AKR-manifold seed. It was added because the
upstream trajectory optimizer constrains the appliance anchor at the endpoint
and then filters intermediate anchor drift after optimization. Generating the
seed on the articulation manifold makes the fixed-anchor constraint explicit
throughout the path; this is not an asset-specific controller.

### Current evidence and boundary

For RealAppliance `055`, joint `/World/part_03/part_03`, fresh GraspGen rank 0:

- 600 native candidates were generated; rank 0 came from the fixed handle body
  with confidence `0.9851905703544617`;
- the target is `72 deg`, i.e. 80% of the 90-degree source range;
- all 32 articulation samples had collision-feasible IK (148--217 solutions per
  layer);
- the selected path is feasible at all 32 samples;
- maximum appliance-anchor position drift is `1.97e-5 m`, and rotation drift is
  `0 rad`.

The generic fixed-Franka staging search then evaluated 216 root poses over a
robot-relative translation/yaw grid. Its best pose produced a 32/32 feasible
manifold path (44--202 collision-feasible IK solutions per layer) with maximum
anchor drift `1.674e-5 m`. The physical executor consumes only the seven Franka
columns from that path; the eighth AKR/object column is never sent to PhysX.

## Paper-version reproducibility

The paper stack is being retained separately from the Isaac Sim 4.5
compatibility executor. Submodules must be checked out at the commits pinned by
upstream AutoMoMa rather than replaced by convenient local versions:

- IsaacLab-Arena: `dc1a4c4a0f6e6d4a6c680d515db034083901074c`;
- cuRobo: `2015f966fcc4ffacba73d61800770b5e5b373ffe`;
- RoboTwin: `c754b5a27e173e63b8023d1e6b71b17fd8238be9`;
- LeRobot: `3a4f0f832c22b20c441a1f65348a5177a04667f3`.

Upstream currently declares Isaac Sim 5.1, Python 3.11, CUDA toolkit 12.8,
PyTorch 2.7.0/cu128, and Warp 1.12.1. Results obtained with the installed Isaac
Sim 4.5 compatibility executor are explicitly labelled as such and are not
presented as paper-stack reproduction evidence.

This is planning evidence only. `strict_physical_success` remains false until
Isaac/PhysX executes robot DOFs only and the contact and penetration audits pass.
No native success video exists yet and none should be inferred from the planner
metrics.
