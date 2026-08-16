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

The public repository consumes grasp-specific AKR files but does not publish
the generator or runnable assets. This branch therefore supplies the missing
mechanical adapter: it appends `EE -> handle -> inverse target joint -> fixed
object root` to the G2 chain. The AKR coordinate is
`q_akr = -(q_object - q_initial)`, so keeping the AKR terminal object-root pose
fixed makes the optimized base/body/arm follow the desired articulation.

The adapter is responsible for:

1. exposing G2 planar base, body, and one selected arm as a cuRobo cspace;
2. conservatively fitting planner collision spheres to the G2 base visual bounds;
3. generating both left- and right-hand hypotheses;
4. converting RealAppliance articulation metadata into an AutoMoMa task;
5. executing the robot-only trajectory in Isaac Sim drive mode;
6. enforcing the stricter physical-success contract.

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

The compatibility environment on the 4090 is
`/home/zhiyuan/external_first/envs/automoma_plan310`. It uses Python 3.10,
Torch 2.5.1+cu118, NumPy 1.26.4, and Warp 1.7.2. The Warp pin matters: the
newer 1.16 wheel removed the API expected by AutoMoMa's pinned cuRobo commit.
Top-level compatibility pins live in `configs/g2_plan_compat_constraints.txt`.

## Automatic task extraction

`extract_usd_tasks()` resolves all task-compatible parts from each asset's
`gt_part.json`, then follows the USD `physics:body0`/`physics:body1`
relationships to the authored revolute or prismatic joint. It normalizes
revolute degrees to radians and verifies that the two authored local joint
frames agree in world coordinates. The resulting contract contains the target
link, object-root link, limits, reset state, world pivot, world axis, and the
initial transforms required by AKR. No asset ID is inspected.

For a candidate EE grasp pose, the adapter computes
`EE -> target link -> initial joint frame -> object root`. The inserted target
joint has the opposite displacement. A unit test checks the actual transform
invariant over several door angles; merely loading the generated URDF is not
treated as sufficient evidence.

## Collision-aware AKR planning status

The diagnostic planner now consumes the existing learned grasp-candidate pool,
places the complete appliance collision geometry in its authored world pose,
removes only the moving target cluster from the fixed planning world, and
optimizes both hands and SE(2) base motion. Candidate rank is output metadata,
not a required runtime input. Among valid trajectories, selection minimizes a
single weighted cspace arc-length objective; the base is a soft global cost and
has no fixed movement share.

The same code found valid collision-aware plans for assets 020, 038, 039, and
055. Both hands failed collision checking for 057, which is retained as a hard
failure case rather than receiving an asset-specific exception. These are only
kinematic planning results and must not be reported as physical success.

## Physical execution status

`tools/g2/execute_g2_akr_physx.py` independently executes the selected AKR path
with G2 joint drives and the four physical swerve modules. It starts at the
selected contact pose for this gate, closes the real SwiftPicker mechanism, and
leaves the appliance articulation passive throughout. The target joint is not
written, no attachment is created, and target collisions remain enabled.

Contact success is measured near the selected learned handle surface after
transforming its points with the live moving body. Contact elsewhere on the
door is logged separately and cannot satisfy the grasp contract. High friction
is applied to the fingers; when a handle is not a separate rigid body, it is
not applied to the complete door. All three cameras use twice the original
stand-off distance.

The first auditable 055 execution reached 18.423% physical opening with
selected-surface contact, 7.798 N maximum force, 0.140 mm maximum penetration,
and zero off-selected-surface force. It then lost contact because the original
fraction-based progress gate let the planned contact point lead the physical
door by roughly 4--5 cm. A metric-gated run limited this lead to 3 mm and kept
bilateral contact, but stalled at 1.147% because 3 mm preload did not overcome
static friction and the 5 mm base tracking gate was marginally too tight.

The controller therefore uses a contact-conditioned, metric probe:

- ordinary planned-contact lead is at most 3 mm;
- after 0.5 s with selected contact and no measurable articulation progress,
  the limit ramps continuously over 1.0 s to at most 15 mm;
- any target progress or contact loss immediately resets the probe;
- free-space base tracking tolerance is 8 mm and selected-contact-loaded
  tolerance is 15 mm, while yaw and manipulator tracking remain independently
  gated.

This probe is a global static-friction mechanism, not an asset rule. Subsequent
055 ablations isolated three separate effects:

- the original progress statistic included a close-phase transient; after
  separating opening-only progress, the nominal controller advanced only
  0.028%;
- scaling every authored planner drive effort by 1.5 reduced the saturated
  body-joint tracking error from 0.057 rad to 0.0055 rad and produced 1.519%
  physical opening;
- allowing bounded 15 mm base compliance under selected contact produced
  8.292% opening with 0.345 mm peak penetration, after which the single-finger
  contact slipped and the passive door closed again;
- doubling the base position gain from 1.5 to 3.0 made the result worse:
  opening fell to 6.639%, off-selected contact rose to 49.47 N, and penetration
  rose to 1.989 mm. The default gain was therefore reverted rather than being
  retained as an unexplained tuning knob.

These are failure cases, not successes. They show that further force/gain
tuning of one trajectory is no longer the correct search direction.

## Automatic physical hypothesis search

`tools/g2/search_g2_akr_physx.py` removes the remaining manual choice of grasp
rank, hand, and AKR path. It expands every successful planning attempt into
physical trials and executes them under a bounded budget. Search is breadth
first over hypotheses: the best path of every learned contact/hand hypothesis
is tested before any hypothesis consumes budget on its second-best path. The
exact launch command, hypothesis metadata, result path, progress, and failure
reasons are persisted before and after every Isaac invocation. A process guard
either stops immediately or waits with an explicit timeout when the separate
G2 pipeline occupies Isaac; it never kills or preempts that work.

For the first 055 pool, the top five learned contacts combined with both hands
gave 10/10 collision-valid planning hypotheses and 52 physical path trials.
This pool is intentionally broader than the earlier single right-hand,
rank-zero, 20%-depth diagnostic. The first physical breadth round tested the
best AKR path for ranks zero through two on both hands:

| rank | hand | maximum opening | selected contact during opening | key observation |
|---:|:---:|---:|:---:|:---|
| 0 | left | 0.326% | no | high off-selected contact |
| 0 | right | 5.935% | yes | contact later lost |
| 1 | left | 18.673% | yes | clean contact; only interaction-lead gate remained |
| 1 | right | 1.743% | no | no target contact acquired |
| 2 | left | 7.096% | yes | interaction lead reached the probe limit |
| 2 | right | 1.743% | no | no target contact acquired |

None is a strict physical success. The rank-one left trial is nevertheless a
useful controller diagnostic: it had zero off-selected force, 1.083 mm peak
penetration, and 32.997 N peak force. Replaying the automatically selected best
physical trial with a 4 mm ordinary lead limit produced 18.285%, slightly worse
than the 3 mm run, so the relaxed limit was rejected. The replay recorded 1,854
selected-contact steps, including 1,197 bilateral-contact steps, and showed
that the remaining stop was a time-budget/interaction-lead issue rather than
contact loss.

The executor now treats `maximum-opening-seconds` as the actual total opening
budget instead of silently ending a short nominal trajectory after only eight
extra seconds. It also terminates a candidate after a one-second initial
contact window when the reference cannot advance without contact; waiting the
old eight-second stall budget could not create contact because AKR progression
is contact-gated.

A 60-second replay of the automatically selected rank-one left path reached
52.140% opening. It accumulated 2,537 selected-contact steps (1,306 bilateral),
remained free of off-selected contact until the late slip, and used 84.2% of
the adaptive lead probe. After the fingers left the handle, the passive door
continued under its own dynamics, while body-joint tracking error grew to
1.011 rad. This is still below the 70% revolute acceptance threshold and is
recorded as a failure, not promoted by its visually substantial partial open.

The first contact-loss recovery experiment exposed an unsafe recovery design.
After 0.1 seconds without selected contact, it opened the gripper to 30% and
continuously remapped the old AKR path to the live articulation. The transient
was detected at 19.673%; while the passive door returned toward closed, the
controller chased the regressing reference to zero progress. Neither of two
regrasp attempts recovered contact, body-joint tracking error reached 1.654
rad, peak contact force reached 384.443 N, and maximum opening was only
20.333%. That policy was rejected.

The replacement is a progress-preserving local reclose rather than a global
path chase:

- contact loss is debounced for 0.2 seconds;
- the measured robot joints and planar base pose are frozen for 0.2 seconds,
  allowing transient contact to return without moving the arm;
- if necessary, the gripper unloads only to 90%, holds locally for 0.2 seconds,
  and recloses over 0.4 seconds;
- the process is bounded to two attempts and aborts if articulation rolls back
  by more than five percentage points;
- only after stable measured handle contact is recovered is AKR progress
  remapped to the live articulation and normal path following resumed.

These values are controller-wide safety limits, not asset-specific offsets.
The local-reclose policy remains experimental until a fresh PhysX replay
confirms that it both avoids the rejected high-force regression and recovers
real handle contact.

Results before the complete home-to-contact and release/retreat phases remain
`dataset_ready=false` even if they pass the physical opening threshold.

## Gates

1. Build and validate left/right G2 planner configs with three virtual base DOFs.
2. Import the pinned cuRobo planner in an isolated CUDA 11.8 environment.
3. Solve an empty-scene left/right/base/arm IK smoke task.
4. Convert one RealAppliance handled-open task without asset-specific code.
5. Plan an AKR trajectory and replay robot commands in physics drive mode.
6. Obtain one strict physical success on a diagnostic asset.
7. Obtain three unadapted assets with one strict success each, then three successes each.

At every gate, a failure is saved with the exact environment, command, traceback, and smallest unresolved interface mismatch.
