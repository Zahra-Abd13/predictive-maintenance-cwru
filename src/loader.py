import os
import scipy.io as sio
import numpy as np


# ─────────────────────────────────────────────
#  LABEL MAP
# 
# ─────────────────────────────────────────────
LABEL_MAP = {
    'normal': 0,
    'ir':     1,   # Inner Race fault
    'ball':   2,   # Ball fault
    'or':     3,   # Outer Race fault
}


def infer_label(filename: str) -> int:
    """
    Infer the fault label from the filename.
    CWRU filenames follow patterns like:
        normal_0.mat, IR007_0.mat, B007_0.mat, OR007@6_0.mat
    Returns an integer label (0–3).
    """
    fname = filename.lower()
    for key, label in LABEL_MAP.items():
        if key in fname:
            return label
    raise ValueError(
        f"Cannot infer label from filename: '{filename}'. "
        f"Expected 'normal', 'ir', 'b0'/'ball', or 'or' in the name."
    )


def load_cwru_mat(filepath: str) -> np.ndarray:
    """
    Load a single CWRU .mat file and return the
    Drive-End (DE) accelerometer signal as a 1-D numpy array.

    Parameters
    ----------
    filepath : str
        Full path to the .mat file.

    Returns
    -------
    np.ndarray
        1-D array of vibration signal values.
    """
    mat = sio.loadmat(filepath)

    # Find the key that ends with '_DE_time' (drive-end sensor)
    de_keys = [k for k in mat.keys() if 'DE_time' in k]

    if len(de_keys) == 0:
        raise KeyError(
            f"No '_DE_time' key found in '{filepath}'.\n"
            f"Available keys: {list(mat.keys())}"
        )

    signal = mat[de_keys[0]].flatten()
    return signal


def build_dataset(data_dir: str) -> list:
    """
    Load all .mat files in data_dir and return a list of records.

    Parameters
    ----------
    data_dir : str
        Path to the folder containing .mat files (e.g. 'data/raw/').

    Returns
    -------
    list of dict, each dict has:
        'signal' : np.ndarray  — raw vibration signal
        'label'  : int         — 0=Normal, 1=IR, 2=Ball, 3=OR
        'file'   : str         — original filename
    """
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Directory not found: '{data_dir}'")

    mat_files = [f for f in os.listdir(data_dir) if f.endswith('.mat')]

    if len(mat_files) == 0:
        raise FileNotFoundError(
            f"No .mat files found in '{data_dir}'. "
            f"Make sure you downloaded the CWRU dataset correctly."
        )

    records = []
    skipped = []

    for fname in sorted(mat_files):
        fpath = os.path.join(data_dir, fname)
        try:
            label  = infer_label(fname)
            signal = load_cwru_mat(fpath)
            records.append({
                'signal': signal,
                'label':  label,
                'file':   fname
            })
        except (ValueError, KeyError) as e:
            skipped.append(fname)
            print(f"[WARNING] Skipped '{fname}': {e}")

    print(f"\n✓ Loaded {len(records)} files successfully.")
    if skipped:
        print(f"  Skipped {len(skipped)} files: {skipped}")

    return records


def get_label_name(label: int) -> str:
    """Convert integer label to human-readable string."""
    names = {0: 'Normal', 1: 'Inner Race', 2: 'Ball', 3: 'Outer Race'}
    return names.get(label, 'Unknown')


def summarize_dataset(records: list) -> None:
    """
    Print a summary of the loaded dataset:
    class distribution, signal lengths, sampling info.
    """
    from collections import Counter
    labels = [r['label'] for r in records]
    counts = Counter(labels)

    print("\n── Dataset Summary ──────────────────────────")
    print(f"  Total files loaded : {len(records)}")
    print(f"  Signal length range: "
          f"{min(len(r['signal']) for r in records):,} – "
          f"{max(len(r['signal']) for r in records):,} samples")
    print("\n  Class distribution:")
    for label, count in sorted(counts.items()):
        pct = count / len(records) * 100
        bar = '█' * count
        print(f"    {label} ({get_label_name(label):12s}): "
              f"{count:3d} files  ({pct:.1f}%)  {bar}")
    print("─────────────────────────────────────────────\n")
