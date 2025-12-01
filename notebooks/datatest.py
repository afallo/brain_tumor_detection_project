from pathlib import Path
import h5py
import numpy as np

here = Path(__file__).resolve().parent
mat_path = here / "1.mat"

print(mat_path)
print(mat_path.exists())  # doit afficher True

with h5py.File(mat_path, "r") as f:
    grp = f["cjdata"]
    print(list(grp.keys()))   # par ex. ['image', 'label', 'PID']
    img = grp["image"][:]
    label = grp["label"][:]
