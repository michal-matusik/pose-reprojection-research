"""Virtual-camera rectification of 2D keypoints via 3D-aware reprojection.

Ported from ../Research/rectify.py (see that repo's rectify.py docstring for the
original spec). The core geometry — IMU/extrinsics-based leveling, RANSAC floor-plane
fit from the feet, yaw-preserving camera axis construction, sequence-constant virtual
camera placement, per-frame reprojection, optional Sav-Gol smoothing and size control
— is unchanged. Two things were generalized so this runs against data this repo
actually produces (RTMPose/COCO-17 + VideoPose3D/H36M-17), instead of only the
Halpe-26 + phone-IMU data the original script assumed:

  * `joint_format`: "coco17" or "halpe26" selects the 2D keypoint layout on both
    input and output; both are converted through canonical H36M-17 internally.
  * `vertical_source`: "imu_pitch_roll" (original IMU pitch/roll input) or
    "camera_extrinsics" (derives the vertical unit vector directly from a known
    world-to-camera rotation matrix, which MPI-INF-3DHP provides via calibration —
    no IMU reading needed).

Hypothesis under test (see project README): a first-pass 3D pose estimate may carry
enough depth information to approximately canonicalize 2D keypoints from an
unfavorable camera viewpoint, by reprojecting them through a leveled virtual camera
before a second 3D lifting pass. This module implements the reprojection step only;
it does not itself claim or measure whether re-lifting improves accuracy — see
scripts/evaluate_rectification_mpi_clip.py for that.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    from scipy.signal import savgol_filter  # type: ignore
except Exception:
    savgol_filter = None

from pose_reprojection.core.joint_maps import (
    COCO17_FROM_H36M17,
    DEFAULT_HALPE26_TO_H36M17,
)


# ------------------------------- Config -------------------------------- #

@dataclass
class RectifyConfig:
    joint_format: str = "coco17"           # "coco17" or "halpe26"
    vertical_source: str = "camera_extrinsics"  # "camera_extrinsics" or "imu_pitch_roll"

    # vertical_source == "imu_pitch_roll"
    imu_pitch_roll_deg: Tuple[float, float] = (0.0, 0.0)
    R_imu_to_cam: Optional[np.ndarray] = None

    # vertical_source == "camera_extrinsics"
    world_up_axis: Tuple[float, float, float] = (0.0, 1.0, 0.0)

    # Virtual camera & distance control
    virtual_cam_height: Optional[float] = None  # h_cam; None -> 0.9 * median body height along n
    depth_factor: float = 1.5
    foreshorten_tau: float = 0.25
    keep_yaw: bool = True
    horizontal_recentering: bool = False

    # Intrinsics / size control
    K_out: Optional[np.ndarray] = None
    size_control_enabled: bool = False
    size_control_target_fill: float = 0.50
    size_control_tolerance: float = 0.05
    size_control_control: str = "adjust_focal"  # or "adjust_depth"
    size_control_bounds: Tuple[float, float] = (0.9, 1.1)

    # Smoothing
    savgol_window: int = 0
    savgol_polyorder: int = 2

    # Output
    coord_system_out: str = "undistorted"

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "RectifyConfig":
        cfg = RectifyConfig()
        for key, value in d.items():
            if not hasattr(cfg, key):
                raise ValueError(f"Unknown rectify config key: {key}")
            if key == "R_imu_to_cam" and value is not None:
                value = np.asarray(value, dtype=np.float64)
            if key == "K_out" and value is not None:
                value = np.asarray(value, dtype=np.float64)
            if key in ("imu_pitch_roll_deg", "world_up_axis", "size_control_bounds") and value is not None:
                value = tuple(float(v) for v in value)
            setattr(cfg, key, value)
        return cfg


def target_index_map(joint_format: str) -> Dict[int, int]:
    """Return {target_2d_joint_index: h36m17_index} for the given 2D joint layout."""
    if joint_format == "coco17":
        return dict(COCO17_FROM_H36M17)
    if joint_format == "halpe26":
        return {
            halpe_idx: h36_idx
            for halpe_idx, h36_idx in enumerate(DEFAULT_HALPE26_TO_H36M17)
            if h36_idx >= 0
        }
    raise ValueError(f"Unknown joint_format: {joint_format}")


def num_target_joints(joint_format: str) -> int:
    return {"coco17": 17, "halpe26": 26}[joint_format]


# --------------------------- Geometry primitives ---------------------------- #
# Unchanged from ../Research/rectify.py.

def _normalize(v: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    n = np.linalg.norm(v)
    if not np.isfinite(n) or n < eps:
        return v * 0.0
    return v / n


def _ensure_unit(n: np.ndarray) -> np.ndarray:
    n = _normalize(n.reshape(3))
    if not np.isfinite(n).all():
        return np.array([0.0, 0.0, 1.0], dtype=np.float64)
    return n


def rot_x(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)


def rot_z(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)


def n_from_imu(pitch_deg: float, roll_deg: float, R_imu_to_cam: Optional[np.ndarray]) -> np.ndarray:
    """Vertical unit vector (in camera frame) from IMU pitch/roll (yaw ignored)."""
    pitch = math.radians(pitch_deg)
    roll = math.radians(roll_deg)

    up0_cam = np.array([0.0, -1.0, 0.0], dtype=np.float64)  # image-y points down

    if R_imu_to_cam is not None:
        up0_imu = R_imu_to_cam.T @ up0_cam
        R_pr_imu = rot_z(roll) @ rot_x(pitch)
        up_imu = R_pr_imu @ up0_imu
        n_cam = R_imu_to_cam @ up_imu
    else:
        R_pr_cam = rot_z(roll) @ rot_x(pitch)
        n_cam = R_pr_cam @ up0_cam

    return _ensure_unit(n_cam)


def n_from_extrinsics(R_world_to_cam: np.ndarray, world_up_axis: Tuple[float, float, float]) -> np.ndarray:
    """Vertical unit vector (in camera frame), derived from a known world->camera
    rotation matrix and a world "up" axis — used when calibrated extrinsics are
    available (e.g. MPI-INF-3DHP) instead of an IMU reading."""
    up_world = _ensure_unit(np.asarray(world_up_axis, dtype=np.float64))
    n_cam = R_world_to_cam @ up_world
    return _ensure_unit(n_cam)


def build_leveled_camera_axes(n: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Construct camera axes (right, down, forward) with yaw preserved:
      * forward0 = [0,0,1]
      * forward = project forward0 onto plane perp to n
      * right   = normalize(cross(forward, n))
      * up      = cross(right, forward); return down = -up
    """
    forward0 = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    f = forward0 - np.dot(forward0, n) * n
    if np.linalg.norm(f) < 1e-9:
        x_axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        f = x_axis - np.dot(x_axis, n) * n
    f = _normalize(f)
    right = _normalize(np.cross(f, n))
    up = _normalize(np.cross(right, f))
    down = -up
    return right, down, f


def rotation_world_to_cam(right: np.ndarray, down: np.ndarray, forward: np.ndarray) -> np.ndarray:
    """World->camera rotation R such that X_cam = R @ (X_world - C)."""
    R = np.stack([right, down, forward], axis=0)
    U, _, Vt = np.linalg.svd(R, full_matrices=False)
    R_ortho = U @ Vt
    if np.linalg.det(R_ortho) < 0:
        U[:, -1] *= -1
        R_ortho = U @ Vt
    return R_ortho


def project_points(K: np.ndarray, R: np.ndarray, C: np.ndarray, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """X: (J,3) world coords; returns (u,v) pixel coords and z_cam depth."""
    X_cam = (R @ (X - C).T).T
    z = X_cam[:, 2]
    uv = np.zeros((X.shape[0], 2), dtype=np.float64)
    mask = np.abs(z) > 1e-9
    uv[mask, 0] = K[0, 0] * (X_cam[mask, 0] / z[mask]) + K[0, 2]
    uv[mask, 1] = K[1, 1] * (X_cam[mask, 1] / z[mask]) + K[1, 2]
    return uv, z


# --------------------------- Floor / height helpers -------------------------- #

def body_height_along(n: np.ndarray, X: np.ndarray) -> float:
    proj = X @ n.reshape(3)
    return float(np.max(proj) - np.min(proj))


def seq_heights_along(n: np.ndarray, X_seq: np.ndarray) -> np.ndarray:
    T = X_seq.shape[0]
    out = np.zeros(T, dtype=np.float64)
    for t in range(T):
        out[t] = body_height_along(n, X_seq[t])
    return out


def seq_depth_spread_along_forward(forward: np.ndarray, X_seq: np.ndarray) -> np.ndarray:
    T = X_seq.shape[0]
    out = np.zeros(T, dtype=np.float64)
    f = forward.reshape(3)
    for t in range(T):
        s = X_seq[t] @ f
        out[t] = float(np.max(s) - np.min(s))
    return out


def fit_floor_plane_from_feet(
    X_seq: np.ndarray,
    n: np.ndarray,
    foot_indices: Tuple[int, int] = (3, 6),  # (RAnkle, LAnkle) in H36M-17
    q_high: float = 25.0,
    ransac_iters: int = 200,
    ransac_thresh: float = 0.02,
) -> Tuple[np.ndarray, float, Dict[str, Any]]:
    """
    Fit a near-horizontal floor plane from foot points across the clip.
    Returns (n_plane, d_plane, stats); plane equation: n_plane . X + d_plane = 0.
    Falls back to height-only (n_plane = n) if RANSAC is unstable or the fitted
    plane deviates too much (>10 deg) from the input vertical.
    """
    pts: List[np.ndarray] = []
    heights: List[float] = []
    for t in range(X_seq.shape[0]):
        X = X_seq[t]
        for j in foot_indices:
            p = X[j]
            heights.append(float(np.dot(p, n)))
            pts.append(p)
    if len(pts) < 8:
        h_med = float(np.median(heights)) if heights else 0.0
        return n, -h_med, dict(method="height_only", inliers=0, total=len(pts), angle_deg=0.0)

    pts_arr = np.vstack(pts)
    heights_arr = np.array(heights, dtype=np.float64)
    h_thresh = float(np.percentile(heights_arr, q_high))
    mask = heights_arr <= h_thresh
    cand = pts_arr[mask]
    if cand.shape[0] < 8:
        h_med = float(np.median(heights_arr))
        return n, -h_med, dict(method="height_only_lowpts", inliers=cand.shape[0], total=len(pts), angle_deg=0.0)

    best_inliers: List[int] = []
    best_n = None
    best_d = None
    rng = np.random.default_rng(12345)
    for _ in range(ransac_iters):
        idx = rng.choice(cand.shape[0], size=3, replace=False)
        A = cand[idx]
        v1 = A[1] - A[0]
        v2 = A[2] - A[0]
        n_try = _normalize(np.cross(v1, v2))
        if np.linalg.norm(n_try) < 1e-9:
            continue
        if np.dot(n_try, n) < 0:
            n_try = -n_try
        d_try = -float(np.dot(n_try, A[0]))
        dist = np.abs(cand @ n_try + d_try)
        inliers = np.nonzero(dist <= ransac_thresh)[0]
        if inliers.size > len(best_inliers):
            best_inliers = inliers.tolist()
            best_n, best_d = n_try, d_try

    if best_n is None or len(best_inliers) < 8:
        h_med = float(np.median(heights_arr))
        return n, -h_med, dict(
            method="height_only_fallback",
            inliers=(len(best_inliers) if best_inliers is not None else 0),
            total=len(pts), angle_deg=0.0,
        )

    P = cand[np.array(best_inliers)]
    mu = np.mean(P, axis=0)
    _, _, Vt = np.linalg.svd(P - mu, full_matrices=False)
    n_ref = _normalize(Vt[-1, :])
    if np.dot(n_ref, n) < 0:
        n_ref = -n_ref
    d_ref = -float(np.dot(n_ref, mu))

    angle = float(np.degrees(np.arccos(np.clip(np.dot(_normalize(n), _normalize(n_ref)), -1.0, 1.0))))

    if angle > 10.0:
        h_med = float(np.median(heights_arr))
        return n, -h_med, dict(method="height_only_angle_guard", inliers=len(best_inliers), total=len(pts), angle_deg=angle)

    return n_ref, d_ref, dict(method="ransac_refined", inliers=len(best_inliers), total=len(pts), angle_deg=angle)


# ------------------------------ Size control -------------------------------- #

def compute_fill_height(uv_joints: np.ndarray) -> float:
    y_min = float(np.min(uv_joints[:, 1]))
    y_max = float(np.max(uv_joints[:, 1]))
    return max(1.0, y_max - y_min)


def adjust_focal_scale_for_fill(
    H_img: float,
    uv_joints: np.ndarray,
    target_fill: float,
    tol: float,
    s_bounds: Tuple[float, float],
) -> float:
    subj_h = compute_fill_height(uv_joints)
    fill = subj_h / max(1.0, H_img)
    if abs(fill - target_fill) <= tol:
        return 1.0
    s = target_fill / max(fill, 1e-9)
    return float(np.clip(s, s_bounds[0], s_bounds[1]))


# ------------------------------ Main pipeline -------------------------------- #

def rectify_sequence(
    target_2d: np.ndarray,          # (N, J, 3) target-format keypoints [u, v, conf]
    X_seq: np.ndarray,               # (N, 17, 3) pelvis-centered H36M-17 3D (scale-free)
    K_base: np.ndarray,              # (3,3) camera intrinsics
    image_size: Tuple[int, int],     # (W, H)
    cfg: RectifyConfig,
    R_world_to_cam: Optional[np.ndarray] = None,  # required if vertical_source == "camera_extrinsics"
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Reproject pelvis-centered 3D through a leveled, yaw-preserving virtual
    camera and write the result back into `target_2d`'s joint layout.

    Returns (rectified_2d, metadata) where rectified_2d has the same shape as
    target_2d; joints with no H36M-17 source mapping are left unchanged.
    """
    N = target_2d.shape[0]
    if X_seq.shape[0] != N:
        raise ValueError(f"Length mismatch: 2D frames={N}, 3D frames={X_seq.shape[0]}")

    target_from_h36 = target_index_map(cfg.joint_format)

    K_base = cfg.K_out.copy() if (cfg.K_out is not None) else K_base.copy()
    H_img = float(image_size[1])

    if cfg.vertical_source == "imu_pitch_roll":
        pitch_deg, roll_deg = cfg.imu_pitch_roll_deg
        n_input = n_from_imu(pitch_deg, roll_deg, cfg.R_imu_to_cam)
    elif cfg.vertical_source == "camera_extrinsics":
        if R_world_to_cam is None:
            raise ValueError("vertical_source='camera_extrinsics' requires R_world_to_cam")
        n_input = n_from_extrinsics(R_world_to_cam, cfg.world_up_axis)
    else:
        raise ValueError(f"Unknown vertical_source: {cfg.vertical_source}")

    n_plane, d_plane, plane_stats = fit_floor_plane_from_feet(X_seq, n_input)
    n = n_plane

    if cfg.keep_yaw:
        right, down, forward = build_leveled_camera_axes(n)
    else:
        right = np.array([1.0, 0.0, 0.0])
        down = np.array([0.0, 1.0, 0.0])
        forward = np.array([0.0, 0.0, 1.0])
    R = rotation_world_to_cam(right, down, forward)

    H_seq = seq_heights_along(n, X_seq)
    Dz_seq = seq_depth_spread_along_forward(forward, X_seq)

    median_H = float(np.median(H_seq)) if H_seq.size else 1.0
    p95_Dz = float(np.percentile(Dz_seq, 95)) if Dz_seq.size else 0.0
    d_from_H = cfg.depth_factor * max(1e-6, median_H)
    d_from_tau = (p95_Dz / max(cfg.foreshorten_tau, 1e-6)) if p95_Dz > 0 else 0.0
    d = max(d_from_H, d_from_tau, 1e-4)

    if cfg.virtual_cam_height is not None:
        h_cam = float(cfg.virtual_cam_height)
        vh_source = "config"
    else:
        h_cam = 0.9 * median_H
        vh_source = "0.9x_median_body_height"

    beta = h_cam - d_plane
    C_base = beta * n - d * forward

    rect_2d = target_2d.copy()
    stats_behind = 0
    scales_used: List[float] = []

    for t in range(N):
        X = X_seq[t]
        C = C_base.copy()

        if cfg.horizontal_recentering:
            X_cam_root = (R @ (X[0] - C).reshape(3, 1)).reshape(3)
            gamma = float(X_cam_root[0])
            C = C + gamma * right

        K_t = K_base.copy()
        uv_h36, z = project_points(K_t, R, C, X)

        if cfg.size_control_enabled:
            subj_h_px = compute_fill_height(uv_h36)
            fill = subj_h_px / max(1.0, H_img)

            if cfg.size_control_control == "adjust_focal":
                s_t = adjust_focal_scale_for_fill(
                    H_img, uv_h36, cfg.size_control_target_fill, cfg.size_control_tolerance, cfg.size_control_bounds
                )
                if abs(s_t - 1.0) > 1e-6:
                    K_t[0, 0] *= s_t
                    K_t[1, 1] *= s_t
                    uv_h36, z = project_points(K_t, R, C, X)
                scales_used.append(float(s_t))
            elif cfg.size_control_control == "adjust_depth":
                ratio = float(np.clip(fill / max(cfg.size_control_target_fill, 1e-9), 0.9, 1.1))
                d_t = d * ratio
                C = beta * n - d_t * forward
                if cfg.horizontal_recentering:
                    X_cam_root = (R @ (X[0] - C).reshape(3, 1)).reshape(3)
                    gamma = float(X_cam_root[0])
                    C = C + gamma * right
                uv_h36, z = project_points(K_t, R, C, X)
                scales_used.append(1.0)
            else:
                scales_used.append(1.0)
        else:
            scales_used.append(1.0)

        for target_idx, h36_idx in target_from_h36.items():
            u, v = uv_h36[h36_idx]
            zz = z[h36_idx]
            conf = rect_2d[t, target_idx, 2]
            if not np.isfinite(u) or not np.isfinite(v) or zz <= 1e-9:
                rect_2d[t, target_idx, 2] = 0.0
                stats_behind += 1
            else:
                rect_2d[t, target_idx, 0] = float(u)
                rect_2d[t, target_idx, 1] = float(v)
                rect_2d[t, target_idx, 2] = float(conf)

    if cfg.savgol_window and cfg.savgol_window > 0:
        if savgol_filter is not None and cfg.savgol_window % 2 == 1 and cfg.savgol_window >= 3:
            J = rect_2d.shape[1]
            for j in range(J):
                for c in range(2):
                    rect_2d[:, j, c] = savgol_filter(
                        rect_2d[:, j, c],
                        window_length=cfg.savgol_window,
                        polyorder=min(cfg.savgol_polyorder, cfg.savgol_window - 1),
                        mode="interp",
                    )
        # else: silently skip if SciPy not present or bad window (matches original script)

    meta = {
        "frames": N,
        "joint_format": cfg.joint_format,
        "vertical_source": cfg.vertical_source,
        "points_behind_camera": int(stats_behind),
        "coord_system": cfg.coord_system_out,
        "K_base": K_base.tolist(),
        "size_control": dict(
            enabled=cfg.size_control_enabled,
            control=cfg.size_control_control,
            target_fill=cfg.size_control_target_fill,
            tolerance=cfg.size_control_tolerance,
            bounds=list(cfg.size_control_bounds),
            scales_used_stats=dict(
                mean=float(np.mean(scales_used)) if scales_used else 1.0,
                median=float(np.median(scales_used)) if scales_used else 1.0,
                p5=float(np.percentile(scales_used, 5)) if scales_used else 1.0,
                p95=float(np.percentile(scales_used, 95)) if scales_used else 1.0,
            ),
        ),
        "leveling": dict(
            n=list(map(float, n.tolist())),
            plane_method=plane_stats.get("method", ""),
            plane_inliers=plane_stats.get("inliers", 0),
            plane_total=plane_stats.get("total", 0),
            plane_angle_err_deg=float(plane_stats.get("angle_deg", 0.0)),
            d_plane=float(d_plane),
        ),
        "virtual_camera": dict(
            keep_yaw=bool(cfg.keep_yaw),
            horizontal_recentering=bool(cfg.horizontal_recentering),
            h_cam=float(h_cam),
            h_cam_source=vh_source,
            d=float(d),
            forward=list(map(float, forward.tolist())),
            right=list(map(float, right.tolist())),
        ),
        "sequence_stats": dict(
            median_H=float(median_H),
            p95_Dz=float(p95_Dz),
            tau=float(cfg.foreshorten_tau),
            Dz_over_d_p95=float((p95_Dz / d) if d > 0 else 0.0),
        ),
        "notes": (
            "Leveling + floor-from-feet + yaw-preserving virtual camera reprojection. "
            "Constant h_cam & d across the clip; K_base constant unless size control enabled."
        ),
    }

    return rect_2d, meta
