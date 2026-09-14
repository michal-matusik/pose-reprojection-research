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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt", type=Path, default=Path("outputs/eval/mpi_s1_seq1_cam0_frames0_242_gt.npz"))
    parser.add_argument("--pred3d", type=Path, default=Path("outputs/videopose3d_mpi_clip_fullseq/video0_first243_videopose3d_fullseq.npz"))
    parser.add_argument("--pred2d", type=Path, default=Path("outputs/rtmlib_mpi_clip/video0_first243_rtmpose.npz"))
    parser.add_argument("--name", type=str, default="rtmpose_raw_videopose3d")
    parser.add_argument("--output", type=Path, default=Path("outputs/eval/baseline_mpi_s1_seq1_cam0_243.json"))
    args = parser.parse_args()

    gt_data = np.load(args.gt)
    gt3d = gt_data["h36m17_3d_m"].astype(np.float64)
    gt2d = gt_data["h36m17_2d_pixels"].astype(np.float64)
    gt_frames = gt_data["frame_indices"]

    pred3d = load_pred3d(args.pred3d)

    pred2d_data = np.load(args.pred2d)
    pred2d_coco = pred2d_data["keypoints"].astype(np.float32)
    pred2d_h36m = coco17_to_h36m17_seq(pred2d_coco).astype(np.float64)
    pred_frames = pred2d_data["frame_indices"]

    T = min(len(gt3d), len(pred3d), len(pred2d_h36m))
    gt3d = gt3d[:T]
    gt2d = gt2d[:T]
    pred3d = pred3d[:T]
    pred2d_h36m = pred2d_h36m[:T]

    result = {
        "clip": {
            "dataset": "MPI-INF-3DHP",
            "subject": "S1",
            "sequence": "Seq1",
            "camera": int(gt_data["cam_idx"]),
            "num_frames": int(T),
            "gt_frame_start": int(gt_frames[0]),
            "gt_frame_end": int(gt_frames[T - 1]),
            "pred_frame_start": int(pred_frames[0]),
            "pred_frame_end": int(pred_frames[T - 1]),
            "image_size": gt_data["image_size"].astype(int).tolist(),
        },
        "methods": {
            args.name: {
                **summarize_3d(pred3d, gt3d),
                **summarize_2d(pred2d_h36m, gt2d),
            }
        },
        "notes": [
            "MPJPE numbers are only meaningful after joint order, scale, and coordinate-frame checks.",
            "PA-MPJPE/P-MPJPE is the safest first comparison because it allows similarity alignment.",
            "Current VideoPose3D checkpoint was trained for Human3.6M-style 2D detections, so domain mismatch is expected.",
        ],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(json.dumps(result, indent=2))
    print("saved:", args.output)

if __name__ == "__main__":
    main()
