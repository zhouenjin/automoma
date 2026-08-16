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
5. use the selected handle body as the grasp target and every other component
   body as scene-collision context;
6. construct a grasp-specific reversed AKR chain from the USD parent/child
   anchors and rotations;
7. inherit AutoMoMa's adjacent-payload collision topology while preserving
   all other robot and component collision checks;
8. sample the target articulation from closed to the requested fraction, solve
   collision-feasible IK at every sample, and select a continuous path through
   those IK layers;
9. pass the resulting robot-only path to the physical Isaac executor and audit
   contact, penetration, object-joint writes, and attachments.

The RealAppliance adapter also performs a pre-episode passive-stability
calibration. It tests an ascending, asset-independent list of PhysX joint
friction coefficients and retains the smallest value that keeps the closed
joint below the 2.5% drift threshold. Reset writes used by this calibration are
recorded separately; the manipulation episode still forbids object-joint target
commands. Contact sensing and contact material assignment cover every rigid body
in the selected moving component (including fixed handle descendants), not just
the joint's child body.

Strict execution additionally requires finger contact with the exact rigid body
that produced the selected GraspGen candidate. Incidental forearm/door contact
does not satisfy that check. PhysX contact separations are recorded at every
step and the run fails the penetration audit if the maximum depth exceeds 5 mm.
An optional fast audit stops after gripper closing when no finger has touched
the selected interaction body or when the 5 mm threshold is already exceeded.
This reduces failed-candidate runtime without relaxing the success definition.

The compatibility executor explicitly converts the cuRobo `panda_hand` grasp
frame into Isaac Sim 4.5 RMPFlow's synthetic `right_gripper` frame. The latter is
100 mm along hand Z and rotated 180 degrees around hand Z. Passing the hand pose
directly to RMPFlow is invalid and previously produced a 12.99 mm arm/component
penetration; that run is retained as a failure case, not success evidence.

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

Strict physical replay rejected all three initially tested candidates:

- rank 0 reached only `32.17%`, had zero finger/handle contact during opening,
  and penetrated the appliance by `16.75 mm`;
- rank 2 reached only `1.26%`, had zero finger/handle contact, and penetrated by
  `10.00 mm`;
- rank 1 eventually reached `100%` with zero robot/component contact for the
  entire episode, so it was passive opening over the long horizon rather than
  robot manipulation.

All three are representative failures, not dataset-ready episodes. GraspGen had
scored the target geometry in isolation, so high confidence did not exclude a
palm or finger collision with the adjacent door. The native generator now calls
GraspGen's official point-cloud scene collision filter before IK. In the first
`055` rerun, `33/200` handle grasps survived a `3 mm` threshold; the rejected
167 candidates no longer consume AKR or PhysX budget.

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

The exact cuRobo, IsaacLab-Arena, and LeRobot source revisions are present on
the 3090 host. LeRobot was checked out with Git LFS smudging disabled so source
and metadata can be reproduced without downloading training payloads. The
pinned RoboTwin URL names the authors' non-public `chang-xinhai/RoboTwin` fork
and currently returns `Repository not found`; a public revision is not silently
substituted and reported as exact. Both available GPU hosts expose NVIDIA 535
drivers, below the upstream recommendation for the Isaac Sim 5.1 camera stack,
so exact-source retention and Isaac 4.5 compatibility execution remain separate.

No strict native success exists yet. `strict_physical_success` remains false
until Isaac/PhysX executes robot DOFs only and the contact and penetration
audits pass; planner feasibility alone is never promoted to dataset-ready data.
metrics.
