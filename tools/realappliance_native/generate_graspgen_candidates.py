#!/usr/bin/env python3
"""Generate fresh AutoMoMa grasp files through the official GraspGen service."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import msgpack
import msgpack_numpy
import numpy as np
import yaml
import zmq
from scipy.spatial.transform import Rotation


msgpack_numpy.patch()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--component-cloud", type=Path, required=True)
    parser.add_argument("--gripper-description", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5556)
    parser.add_argument("--num-grasps", type=int, default=500)
    parser.add_argument("--topk", type=int, default=200)
    parser.add_argument("--minimum-body-points", type=int, default=256)
    return parser.parse_args()


def request(socket, payload):
    socket.send(msgpack.packb(payload, use_bin_type=True))
    response = msgpack.unpackb(socket.recv(), raw=False)
    if "error" in response:
        raise RuntimeError(response["error"])
    return response


def matrix_to_pose(matrix: np.ndarray) -> np.ndarray:
    quaternion_xyzw = Rotation.from_matrix(matrix[:3, :3]).as_quat()
    return np.asarray([*matrix[:3, 3], *np.roll(quaternion_xyzw, 1)], dtype=np.float32)


def main() -> None:
    args = parse_args()
    cloud = np.load(args.component_cloud)
    points = np.asarray(cloud["points_component_m"], dtype=np.float32)
    body_index = np.asarray(cloud["body_index"], dtype=np.int32)
    body_paths = [str(value) for value in cloud["body_paths"]]
    gripper = yaml.safe_load(args.gripper_description.read_text(encoding="utf-8"))

    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.setsockopt(zmq.RCVTIMEO, 600_000)
    socket.setsockopt(zmq.SNDTIMEO, 600_000)
    socket.connect(f"tcp://{args.host}:{args.port}")
    server = request(socket, {"action": "metadata"})

    inputs = [("component", points)]
    for index, body_path in enumerate(body_paths):
        body_points = points[body_index == index]
        if len(body_points) >= args.minimum_body_points:
            inputs.append((body_path, body_points))
    records = []
    for source, inference_points in inputs:
        response = request(
            socket,
            {
                "action": "infer",
                "point_cloud": inference_points,
                "grasp_threshold": -1.0,
                "num_grasps": args.num_grasps,
                "topk_num_grasps": args.topk,
                "min_grasps": 20,
                "max_tries": 6,
                "remove_outliers": True,
            },
        )
        poses = np.asarray(response["grasps"], dtype=np.float32)
        confidence = np.asarray(response["confidences"], dtype=np.float32).reshape(-1)
        for pose, score in zip(poses, confidence):
            if pose.shape != (4, 4) or not np.isfinite(pose).all() or not np.isfinite(score):
                continue
            records.append((float(score), source, pose))
    records.sort(key=lambda value: value[0], reverse=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    candidates = []
    for rank, (score, source, matrix) in enumerate(records):
        pose = matrix_to_pose(matrix)
        np.save(args.output_dir / f"{rank:04d}.npy", pose)
        candidates.append(
            {
                "rank": rank,
                "source_component": source,
                "graspgen_confidence": score,
                "component_to_gripper_base_pose": pose.tolist(),
            }
        )
    metadata = {
        "schema_version": "automoma.realappliance.graspgen_candidates.v1",
        "provenance": {
            "pipeline": "automoma_native",
            "g2_inputs_used": False,
            "generator": "GraspGen",
        },
        "server": server,
        "gripper": {
            "name": args.gripper_description.stem,
            "width": gripper.get("width"),
            "depth": gripper.get("depth"),
        },
        "component_cloud": str(args.component_cloud),
        "candidate_count": len(candidates),
        "candidates": candidates,
    }
    (args.output_dir / "candidates.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(json.dumps({"output_dir": str(args.output_dir), "candidate_count": len(candidates)}))


if __name__ == "__main__":
    main()
