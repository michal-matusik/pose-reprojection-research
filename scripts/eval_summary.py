"""Shared MPI-INF-3DHP metric summaries, used by both the raw baseline eval
(evaluate_videopose3d_mpi_clip.py) and the rectification ablation
(evaluate_rectification_mpi_clip.py) so the two are directly comparable.
"""
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pose_reprojection.core.joint_maps import BODY14, coco17_to_h36m17

from pose3d_metrics import (
    root_center,
    mpjpe,
    n_mpjpe,
    pa_mpjpe,
    pck_3d,
    auc_3d,
    acceleration_error,
    batch_procrustes_align,
)

# Backward-compatible alias: evaluate_videopose3d_mpi_clip.py originally defined
# this itself as coco17_to_h36m17_seq; kept as a name so existing call sites don't churn.
coco17_to_h36m17_seq = coco17_to_h36m17


def velocity_px(kpts):
    if kpts.shape[0] < 2:
        return np.zeros((0, kpts.shape[1]), dtype=np.float32)
    return np.linalg.norm(np.diff(kpts, axis=0), axis=-1)


def summarize_3d(pred, gt):
    pred_root = root_center(pred, root=0)
    gt_root = root_center(gt, root=0)

    pred14 = pred_root[:, BODY14]
    gt14 = gt_root[:, BODY14]

    pred_pa = batch_procrustes_align(pred_root, gt_root)
    pred14_pa = pred_pa[:, BODY14]

    return {
        "root_mpjpe_all17_mm": float(mpjpe(pred_root, gt_root) * 1000.0),
        "root_mpjpe_body14_mm": float(mpjpe(pred14, gt14) * 1000.0),
        "n_mpjpe_all17_mm": float(n_mpjpe(pred_root, gt_root) * 1000.0),
        "n_mpjpe_body14_mm": float(n_mpjpe(pred14, gt14) * 1000.0),
        "pa_mpjpe_all17_mm": float(pa_mpjpe(pred_root, gt_root) * 1000.0),
        "pa_mpjpe_body14_mm": float(pa_mpjpe(pred14, gt14) * 1000.0),
        "pck150_root_all17": float(pck_3d(pred_root, gt_root, threshold=0.150)),
        "pck150_pa_all17": float(pck_3d(pred_pa, gt_root, threshold=0.150)),
        "pck150_pa_body14": float(pck_3d(pred14_pa, gt14, threshold=0.150)),
        "auc150_pa_all17": float(auc_3d(pred_pa, gt_root, max_threshold=0.150)),
        "accel_error_root_all17_mm_per_frame2": float(acceleration_error(pred_root, gt_root) * 1000.0),
    }


def summarize_2d(pred_2d_h36m, gt_2d_h36m):
    err = np.linalg.norm(pred_2d_h36m - gt_2d_h36m, axis=-1)
    vel = velocity_px(pred_2d_h36m)

    return {
        "mean_2d_error_all17_px": float(np.mean(err)),
        "median_2d_error_all17_px": float(np.median(err)),
        "mean_2d_error_body14_px": float(np.mean(err[:, BODY14])),
        "mean_2d_velocity_all17_px_per_frame": float(np.mean(vel)),
        "max_2d_velocity_all17_px_per_frame": float(np.max(vel)) if vel.size else 0.0,
        "mean_2d_velocity_wrists_px_per_frame": float(np.mean(vel[:, [13, 16]])) if vel.size else 0.0,
        "max_2d_velocity_wrists_px_per_frame": float(np.max(vel[:, [13, 16]])) if vel.size else 0.0,
    }
