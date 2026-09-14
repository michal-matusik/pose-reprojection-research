"""Rectification ablation: does re-lifting virtual-camera-rectified 2D keypoints
change 3D lifting accuracy vs. the raw baseline, on the one MPI-INF-3DHP clip
this repo has ground truth prepared for?

This does NOT run the lifter itself (VideoPose3D requires torch + a downloaded
checkpoint, set up via setup_pose2d_windows.ps1 / download_videopose3d.ps1) — it
assumes you have already produced both a raw baseline and a rectified+relifted
3D prediction, and only compares the two against GT using the same metrics as
scripts/evaluate_videopose3d_mpi_clip.py so the two runs are directly comparable.

Typical end-to-end sequence (each step uses an existing, unmodified script):

  1. (already done)  scripts/extract_rtmlib_keypoints_mpi_clip.py           -> raw 2D
  2. (already done)  scripts/run_videopose3d_fullseq_from_keypoints.py     -> raw 3D
  3.                 scripts/rectify_virtual_camera.py --joint-format coco17
                        --pose2d <raw 2D npz> --pose3d <raw 3D npz>
                        --intrinsics <GT npz> --config configs/rectify/coco17_mpi_default.yaml
                        --out outputs/rectify/<clip>_rectified.npz
                     -> rectified 2D (same npz schema as raw 2D)
  4.                 scripts/run_videopose3d_fullseq_from_keypoints.py
                        --input-keypoints outputs/rectify/<clip>_rectified.npz
                        --output outputs/videopose3d_rectified/<clip>_relifted.npz
                     -> re-lifted 3D
  5.                 python scripts/evaluate_rectification_mpi_clip.py
                        --gt <GT npz> --pred2d-raw <raw 2D npz> --pred3d-raw <raw 3D npz>
                        --pred2d-rectified outputs/rectify/<clip>_rectified.npz
                        --pred3d-rectified outputs/videopose3d_rectified/<clip>_relifted.npz
                     -> comparison JSON with both methods' metrics + deltas

Steps 1-2 are already implemented and were used to produce the smoothing finding
in the README. Steps 3-5 are new. As of writing this script, step 5 has not been
executed end-to-end in this repo (no torch/VideoPose3D checkpoint available in
this environment) -- see README "Status" for what is implemented vs. verified.
"""
from pathlib import Path
import argparse
import json

import numpy as np

from eval_summary import coco17_to_h36m17_seq, summarize_3d, summarize_2d


def load_pred3d(path):
    data = np.load(path)
    pred = data["pred_3d"]
    if pred.ndim == 4:
        pred = pred[0]
    if pred.ndim != 3 or pred.shape[1:] != (17, 3):
        raise ValueError(f"Expected pred_3d as (T, 17, 3), got {pred.shape}")
    return pred.astype(np.float64)


def load_pred2d_h36m(path):
    data = np.load(path)
    coco = data["keypoints"].astype(np.float32)
    return coco17_to_h36m17_seq(coco).astype(np.float64)


def evaluate_one(pred3d_path, pred2d_path, gt3d, gt2d, T):
    pred3d = load_pred3d(pred3d_path)[:T]
    pred2d_h36m = load_pred2d_h36m(pred2d_path)[:T]
    return {
        **summarize_3d(pred3d, gt3d),
        **summarize_2d(pred2d_h36m, gt2d),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gt", type=Path, default=Path("outputs/eval/mpi_s1_seq1_cam0_frames0_242_gt.npz"))
    parser.add_argument("--pred2d-raw", type=Path, required=True)
    parser.add_argument("--pred3d-raw", type=Path, required=True)
    parser.add_argument("--pred2d-rectified", type=Path, required=True)
    parser.add_argument("--pred3d-rectified", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/eval/rectification_ablation_mpi_s1_seq1_cam0.json"))
    args = parser.parse_args()

    gt_data = np.load(args.gt)
    gt3d = gt_data["h36m17_3d_m"].astype(np.float64)
    gt2d = gt_data["h36m17_2d_pixels"].astype(np.float64)

    T = min(
        len(gt3d),
        len(load_pred3d(args.pred3d_raw)),
        len(load_pred2d_h36m(args.pred2d_raw)),
        len(load_pred3d(args.pred3d_rectified)),
        len(load_pred2d_h36m(args.pred2d_rectified)),
    )
    gt3d, gt2d = gt3d[:T], gt2d[:T]

    raw = evaluate_one(args.pred3d_raw, args.pred2d_raw, gt3d, gt2d, T)
    rectified = evaluate_one(args.pred3d_rectified, args.pred2d_rectified, gt3d, gt2d, T)
    delta = {
        k: (rectified[k] - raw[k])
        for k in raw
        if isinstance(raw[k], (int, float)) and isinstance(rectified.get(k), (int, float))
    }

    result = {
        "clip": {
            "dataset": "MPI-INF-3DHP",
            "subject": "S1",
            "sequence": "Seq1",
            "camera": int(gt_data["cam_idx"]) if "cam_idx" in gt_data.files else None,
            "num_frames": int(T),
            "image_size": gt_data["image_size"].astype(int).tolist(),
        },
        "methods": {
            "raw_videopose3d": raw,
            "rectified_videopose3d": rectified,
        },
        "delta_rectified_minus_raw": delta,
        "notes": [
            "Single clip, single subject/sequence/camera -- not a benchmark. Any result here is "
            "preliminary and should not be read as a general claim about the method.",
            "MPJPE numbers are only meaningful after joint order, scale, and coordinate-frame checks.",
            "PA-MPJPE/P-MPJPE is the safest first comparison because it allows similarity alignment.",
            "VideoPose3D checkpoint was trained for Human3.6M-style 2D detections; both raw and "
            "rectified predictions share this domain mismatch, so the delta -- not the absolute "
            "values -- is the relevant comparison for this ablation.",
        ],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(json.dumps(result, indent=2))
    print("saved:", args.output)


if __name__ == "__main__":
    main()
