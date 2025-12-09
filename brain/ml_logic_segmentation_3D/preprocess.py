
import numpy as np
import nibabel as nib
from pathlib import Path

from brain.params import *


## Normalize the images per modality
"""vol: np.ndarray (H, W, D), intensités brutes"""
def normalize_volume(vol):
    mask = vol > 0
    if np.any(mask):
        mean = vol[mask].mean()
        std = vol[mask].std()
        vol[mask] = (vol[mask] - mean) / (std + 1e-8)
    return vol


def crop_or_pad(vol, target_shape):
    H, W, D = vol.shape
    Ht, Wt, Dt = target_shape
    out = np.zeros(target_shape, dtype=vol.dtype)

    h_start = max((Ht - H) // 2, 0)
    w_start = max((Wt - W) // 2, 0)
    d_start = max((Dt - D) // 2, 0)

    h_end = h_start + min(H, Ht)
    w_end = w_start + min(W, Wt)
    d_end = d_start + min(D, Dt)

    vh_start = max((H - Ht) // 2, 0)
    vw_start = max((W - Wt) // 2, 0)
    vd_start = max((D - Dt) // 2, 0)

    vh_end = vh_start + (h_end - h_start)
    vw_end = vw_start + (w_end - w_start)
    vd_end = vd_start + (d_end - d_start)

    out[h_start:h_end, w_start:w_end, d_start:d_end] = vol[vh_start:vh_end,
                                                            vw_start:vw_end,
                                                            vd_start:vd_end]
    return out


def process_case(case_dir: Path):
    vols = []

    # Charger les 4 modalités dans l’ordre défini dans MODALITIES
    for mod_name, pattern in MODALITIES_3D:
        fpath_list = list(case_dir.glob(pattern))
        assert len(fpath_list) == 1, f"Problème pour {mod_name} dans {case_dir}"
        fpath = fpath_list[0]

        nii = nib.load(str(fpath))
        vol = nii.get_fdata().astype(np.float32)
        vol = normalize_volume(vol)
        vol = crop_or_pad(vol, TARGET_SHAPE_3D)
        vols.append(vol)

    # (H,W,D,4)
    img = np.stack(vols, axis=-1).astype(np.float32)

    # Charger le label
    seg_path_list = list(case_dir.glob("*-seg.nii.gz"))
    assert len(seg_path_list) == 1, f"Pas de seg ou multiple seg dans {case_dir}"
    seg_path = seg_path_list[0]

    seg = nib.load(str(seg_path)).get_fdata()
    seg = crop_or_pad(seg, TARGET_SHAPE_3D)
    seg = seg.astype(np.uint8)   # labels {0,1,2,4}
    seg[seg == 4] = 3

    return img, seg
