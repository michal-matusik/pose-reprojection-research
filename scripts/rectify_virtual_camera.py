"""CLI for virtual-camera rectification (see src/pose_reprojection/rectify/virtual_camera.py).

Two joint-format modes, selected by --joint-format or the config's `joint_format` field:

coco17 (this repo's native pipeline: RTMPose 2D + VideoPose3D 3D)
  python scripts/rectify_virtual_camera.py --joint-format coco17 \
    --pose2d outputs/rtmlib_mpi_clip/video0_first243_rtmpose.npz \
    --pose3d outputs/videopose3d_mpi_clip_fullseq/video0_first243_videopose3d_fullseq.npz \
    --intrinsics outputs/eval/mpi_s1_seq1_cam0_frames0_242_gt.npz \
    --config configs/rectify/coco17_mpi_default.yaml \
    --out outputs/rectify/video0_first243_rectified.npz

halpe26 (AlphaPose 2D + MotionBERT 3D, phone-IMU leveling — matches ../Research/rectify.py's I/O)
  python scripts/rectify_virtual_camera.py --joint-format halpe26 \
    --pose2d 2d.json --pose3d video_0.h36m17.npy --intrinsics cam.json \
    --config configs/rectify/halpe26_phone_default.yaml --out 2d_rectified.json
"""
from pathlib import Path
import argparse
import json
import sys

import numpy as np

try:
    import yaml
except Exception:
    yaml = None

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pose_reprojection.core.keypoint_io import load_keypoint_npz, save_keypoint_npz
from pose_reprojection.rectify.virtual_camera import RectifyConfig, rectify_sequence


def _load_yaml_or_json(path):
    path = str(path)
    if path.lower().endswith((".yml", ".yaml")):
        if yaml is None:
            raise RuntimeError("pyyaml is required to load YAML files")
        with open(path, "r") as f:
            return yaml.safe_load(f)
    with open(path, "r") as f:
        return json.load(f)


def load_config(path):
    raw = _load_yaml_or_json(path) if path else {}
    return RectifyConfig.from_dict(raw or {})


def load_intrinsics(path):
    """Loads K (3x3), image size (W,H), and optionally R (3x3 world->camera).

    Accepts either a plain intrinsics file ({"K": ..., "size"/"image_size": ...,
    "R": ...}) or an MPI-INF-3DHP GT npz produced by prepare_mpi_eval_clip.py
    (keys "K", "R", "image_size").
    """
    path = str(path)
    if path.lower().endswith(".npz"):
        data = np.load(path)
        K = np.asarray(data["K"], dtype=np.float64)
        size = tuple(int(v) for v in data["image_size"])
        R = np.asarray(data["R"], dtype=np.float64) if "R" in data.files else None
        return K, size, R

    d = _load_yaml_or_json(path)
    K = np.asarray(d["K"], dtype=np.float64)
    size = d.get("size") or d.get("image_size")
    size = (int(size[0]), int(size[1]))
    R = np.asarray(d["R"], dtype=np.float64) if d.get("R") is not None else None
    return K, size, R


def run_coco17(args, cfg: RectifyConfig):
    pose2d = load_keypoint_npz(args.pose2d)
    keypoints = np.asarray(pose2d["keypoints"], dtype=np.float64)  # (T,17,2)
    scores = np.asarray(pose2d["scores"], dtype=np.float64)        # (T,17)
    frame_indices = pose2d["frame_indices"]

    K, size, R = load_intrinsics(args.intrinsics)

    pred3d = np.load(args.pose3d)["pred_3d"].astype(np.float64)
    if pred3d.ndim == 4:
        pred3d = pred3d[0]
    X_seq = pred3d - pred3d[:, 0:1, :]  # pelvis-center

    target_2d = np.concatenate([keypoints, scores[..., None]], axis=-1)

    rect_2d, meta = rectify_sequence(
        target_2d=target_2d,
        X_seq=X_seq,
        K_base=K,
        image_size=size,
        cfg=cfg,
        R_world_to_cam=R,
    )

    out_data = dict(
        keypoints=rect_2d[..., :2].astype(np.float32),
        scores=rect_2d[..., 2].astype(np.float32),
        frame_indices=frame_indices,
        image_size=np.asarray(pose2d["image_size"]),
    )
    save_keypoint_npz(args.out, out_data)
    return meta


def run_halpe26(args, cfg: RectifyConfig):
    with open(args.pose2d, "r") as f:
        pose2d_list = json.load(f)
    if not isinstance(pose2d_list, list):
        raise ValueError("pose2d must be a JSON list for --joint-format halpe26")

    N = len(pose2d_list)
    target_2d = np.zeros((N, 26, 3), dtype=np.float64)
    for t, item in enumerate(pose2d_list):
        kp = np.asarray(item["keypoints"], dtype=np.float64).reshape(-1, 3)
        if kp.shape != (26, 3):
            raise ValueError(f"pose2d[{t}].keypoints must have length 78 (26*3)")
        target_2d[t] = kp

    K, size, R = load_intrinsics(args.intrinsics)

    if str(args.pose3d).lower().endswith(".npy"):
        X_seq = np.load(args.pose3d).astype(np.float64)
    else:
        with open(args.pose3d, "r") as f:
            X_seq = np.asarray(json.load(f)["keypoints_3d"], dtype=np.float64)
    if X_seq.ndim != 3 or X_seq.shape[1:] != (17, 3):
        raise ValueError(f"pose3d must be (N,17,3), got {X_seq.shape}")
    X_seq = X_seq - X_seq[:, 0:1, :]  # pelvis-center

    rect_2d, meta = rectify_sequence(
        target_2d=target_2d,
        X_seq=X_seq,
        K_base=K,
        image_size=size,
        cfg=cfg,
        R_world_to_cam=R,
    )

    out_list = []
    for t, item in enumerate(pose2d_list):
        new_item = {k: v for k, v in item.items() if k != "keypoints"}
        new_item["keypoints"] = rect_2d[t].reshape(-1).tolist()
        out_list.append(new_item)

    with open(args.out, "w") as f:
        json.dump(out_list, f, indent=2, ensure_ascii=False)
    return meta


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--joint-format", choices=["coco17", "halpe26"], default=None,
                         help="Overrides the config's joint_format if given.")
    parser.add_argument("--pose2d", required=True)
    parser.add_argument("--pose3d", required=True)
    parser.add_argument("--intrinsics", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--out", required=True)
    parser.add_argument("--meta-out", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.joint_format:
        cfg.joint_format = args.joint_format

    if cfg.joint_format == "coco17":
        meta = run_coco17(args, cfg)
    elif cfg.joint_format == "halpe26":
        meta = run_halpe26(args, cfg)
    else:
        raise ValueError(f"Unknown joint_format: {cfg.joint_format}")

    meta_path = args.meta_out or (str(args.out).rsplit(".", 1)[0] + ".meta.json")
    Path(meta_path).parent.mkdir(parents=True, exist_ok=True)
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved rectified 2D ({cfg.joint_format}): {args.out}")
    print(f"Saved metadata: {meta_path}")


if __name__ == "__main__":
    main()
