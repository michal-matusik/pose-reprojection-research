"""Shared joint-layout tables and conversions used by both the evaluation
scripts and the virtual-camera rectifier, so the mapping is defined once.
"""
import numpy as np

H36M17_NAMES = [
    "Pelvis", "RHip", "RKnee", "RAnkle", "LHip", "LKnee", "LAnkle", "Spine1",
    "Neck", "Head", "Site", "LShoulder", "LElbow", "LWrist", "RShoulder",
    "RElbow", "RWrist",
]

# Subset of H36M-17 indices commonly reported as "body14" (drops Spine1/Neck/Site,
# which are less reliably estimated by 2D detectors and 3D lifters alike).
BODY14 = [0, 1, 2, 3, 4, 5, 6, 7, 11, 12, 13, 14, 15, 16]

COCO17_NAMES = [
    "nose", "l_eye", "r_eye", "l_ear", "r_ear",
    "l_shoulder", "r_shoulder", "l_elbow", "r_elbow", "l_wrist", "r_wrist",
    "l_hip", "r_hip", "l_knee", "r_knee", "l_ankle", "r_ankle",
]
COCO = {name: idx for idx, name in enumerate(COCO17_NAMES)}


def _avg(a, b):
    return (a + b) / 2.0


def coco17_to_h36m17(coco_seq):
    """Map a COCO-17 keypoint sequence to the canonical H36M-17 layout.

    coco_seq: (..., 17, C) array (C=2 for pixel coords, C=3 for 3D points).
    Pelvis/thorax/spine/head are derived (COCO has no dedicated pelvis/neck joints).
    """
    coco_seq = np.asarray(coco_seq)
    out = np.zeros(coco_seq.shape[:-2] + (17, coco_seq.shape[-1]), dtype=coco_seq.dtype)

    l_hip = coco_seq[..., COCO["l_hip"], :]
    r_hip = coco_seq[..., COCO["r_hip"], :]
    l_sh = coco_seq[..., COCO["l_shoulder"], :]
    r_sh = coco_seq[..., COCO["r_shoulder"], :]

    pelvis = _avg(l_hip, r_hip)
    thorax = _avg(l_sh, r_sh)
    spine = _avg(pelvis, thorax)

    out[..., 0, :] = pelvis
    out[..., 1, :] = r_hip
    out[..., 2, :] = coco_seq[..., COCO["r_knee"], :]
    out[..., 3, :] = coco_seq[..., COCO["r_ankle"], :]
    out[..., 4, :] = l_hip
    out[..., 5, :] = coco_seq[..., COCO["l_knee"], :]
    out[..., 6, :] = coco_seq[..., COCO["l_ankle"], :]
    out[..., 7, :] = spine
    out[..., 8, :] = thorax
    out[..., 9, :] = thorax
    out[..., 10, :] = coco_seq[..., COCO["nose"], :]
    out[..., 11, :] = l_sh
    out[..., 12, :] = coco_seq[..., COCO["l_elbow"], :]
    out[..., 13, :] = coco_seq[..., COCO["l_wrist"], :]
    out[..., 14, :] = r_sh
    out[..., 15, :] = coco_seq[..., COCO["r_elbow"], :]
    out[..., 16, :] = coco_seq[..., COCO["r_wrist"], :]

    return out


# H36M-17 index that each of the 17 COCO joints "writes to" when rectified 2D is
# projected back out of H36M-17 space. Joints with no direct COCO counterpart
# (pelvis/thorax/spine/head) are NOT invertible from a single H36M joint alone;
# COCO17_FROM_H36M17 gives the best available single-source mapping for writing
# a rectified H36M-17 projection back into COCO-17 slots.
COCO17_FROM_H36M17 = {
    COCO["nose"]: 10,
    COCO["l_shoulder"]: 11,
    COCO["r_shoulder"]: 14,
    COCO["l_elbow"]: 12,
    COCO["r_elbow"]: 15,
    COCO["l_wrist"]: 13,
    COCO["r_wrist"]: 16,
    COCO["l_hip"]: 4,
    COCO["r_hip"]: 1,
    COCO["l_knee"]: 5,
    COCO["r_knee"]: 2,
    COCO["l_ankle"]: 6,
    COCO["r_ankle"]: 3,
}
# l_eye/r_eye/l_ear/r_ear (indices 1-4) have no H36M-17 counterpart and are left
# unmapped (rectifier keeps their original 2D position unchanged).

HALPE26_NAMES = [
    "Nose", "LEye", "REye", "LEar", "REar", "LShoulder", "RShoulder", "LElbow",
    "RElbow", "LWrist", "RWrist", "LHip", "RHip", "LKnee", "RKnee", "LAnkle",
    "RAnkle", "Head", "Neck", "Hip", "LBigToe", "RBigToe", "LSmallToe",
    "RSmallToe", "LHeel", "RHeel",
]

# Default map: for each Halpe-26 index, the H36M-17 index it corresponds to (-1 if none).
# Ported unchanged from ../Research/rectify.py.
DEFAULT_HALPE26_TO_H36M17 = [-1] * 26
_halpe_pairs = [
    (0, 19),  # Pelvis <- Hip
    (1, 12),  # RHip
    (2, 14),  # RKnee
    (3, 16),  # RAnkle
    (4, 11),  # LHip
    (5, 13),  # LKnee
    (6, 15),  # LAnkle
    (8, 18),  # Neck
    (9, 17),  # Head
    (11, 5),  # LShoulder
    (12, 7),  # LElbow
    (13, 9),  # LWrist
    (14, 6),  # RShoulder
    (15, 8),  # RElbow
    (16, 10),  # RWrist
]
for _h36, _halpe in _halpe_pairs:
    DEFAULT_HALPE26_TO_H36M17[_halpe] = _h36
