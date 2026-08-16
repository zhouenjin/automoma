# AutoMoMa-native RealAppliance branch

This branch is the external-first comparison. It starts from upstream AutoMoMa
and uses an AutoMoMa-compatible Franka execution stack. It does not import the
G2 implementation or its controllers.

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
