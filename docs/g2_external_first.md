# G2 + RealAppliance external-first line

## Purpose

This branch starts from AutoMoMa and ports G2/SwiftPicker/RealAppliance into its planning model. It is intentionally independent from the existing G2 repository, which remains the `ours-first` comparison.

The current scope is handled `open`. Handleless opening, `rotate`, `press`, 100-episode collection, and RLDS packaging are outside this gate.

## Non-negotiable outcome contract

- No asset-ID branches, per-asset coordinates, manual offsets, or required candidate ranks.
- Left and right hands are parallel hypotheses. A hinge-side heuristic may be a score feature but cannot select a hand by itself.
- Base SE(2) is optimized with body and arm joints. A geometry-relative seed set is allowed; a fixed front fan is not a feasibility rule.
- The target joint is moved only by measured PhysX contact during the episode.
- No attachment, target-joint writes, target-collision disabling, or set-state replay may be labelled dataset-ready.
- Penetration, robot/target contact, non-target collision, failure reason, and multiview video must be recorded.
- Planning targets are 80% for revolute and 90% for prismatic joints. Strict acceptance is 70% and 80%, respectively.

## External-first boundary

AutoMoMa remains responsible for:

1. AKR construction semantics;
2. batched IK and clustering;
3. cuRobo trajectory optimization;
4. maintaining the end-effector/object relative transform during articulated motion.

The adapter is responsible for:

1. exposing G2 planar base, body, and one selected arm as a cuRobo cspace;
2. generating both left- and right-hand hypotheses;
3. converting RealAppliance articulation metadata into an AutoMoMa task;
4. executing the robot-only trajectory in Isaac Sim drive mode;
5. enforcing the stricter physical-success contract.

Legacy G2 plan selection, fixed base-share controllers, fixed world-frame pulls, and per-asset tuning are not imported.

## Upstream audit findings

Pinned upstream commit: `ec423552bf3ac86ca240062fc0a80bdbbbbd4a63`.

- AutoMoMa's planner does represent base + arm + target object as an AKR chain.
- The public config still lists object metadata, grasp IDs, and goal angles manually.
- The default collision build clears an expanded cuboid around the target object from the ESDF.
- IsaacLab-Arena supports both set-state and drive/physics replay. Only drive/physics is admissible here.
- Upstream final engagement uses a distance threshold; this branch keeps the stricter contact and penetration audit from the G2 project.

## Environment status on the 4090

- Host: Ubuntu 20.04.
- Driver: 535.230.02.
- Existing Isaac Sim: 4.5.0 / Python 3.10 / Torch 2.5.1+cu118.
- Upstream target: Ubuntu 22.04 or 24.04, driver R580, Isaac Sim 5.1, Python 3.11, Torch 2.7+cu128.
- A CUDA 12.8 container fails before startup because driver 535 does not satisfy `cuda>=12.8`.

Therefore the first executable gate is AutoMoMa planning on a separate CUDA 11.8-compatible environment and robot-only physics execution through the existing Isaac Sim 4.5 runtime. A clean 5.1 reproduction remains blocked until the driver is upgraded; it must not be silently claimed as complete.

## Gates

1. Build and validate left/right G2 planner configs with three virtual base DOFs.
2. Import the pinned cuRobo planner in an isolated CUDA 11.8 environment.
3. Solve an empty-scene left/right/base/arm IK smoke task.
4. Convert one RealAppliance handled-open task without asset-specific code.
5. Plan an AKR trajectory and replay robot commands in physics drive mode.
6. Obtain one strict physical success on a diagnostic asset.
7. Obtain three unadapted assets with one strict success each, then three successes each.

At every gate, a failure is saved with the exact environment, command, traceback, and smallest unresolved interface mismatch.

