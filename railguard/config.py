"""Single source of truth for pipeline parameters (see decisions.md D06, D09, D10)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw" / "runs"
OUT_DIR = ROOT / "outputs"

# --- signal / STFT parameters -------------------------------------------------
FS = 2000.0              # Hz, Rail-VIVID acquisition rate
SEG_LEN = 1024           # samples -> 0.512 s, delta_f = 1.953 Hz
HOP = 512                # 50 % overlap
WINDOW = "hann"
DETREND = "constant"     # mean removal only
FMAX = 500.0             # data is low-passed at 500 Hz (D04)
CHANNELS = [1, 2, 3, 4, 5, 6]
SCORE_CHANNELS = [1, 2, 3, 4]   # Ch5/6 are ~10x quieter and <50 Hz only (D05)
FEATURES = ["rms", "dom_freq", "spec_energy", "spec_centroid"]

# --- split (by run, stratified by direction+speed; seed in decisions.md D06) ---
SPLIT_SEED = 42
TRAIN = ["AtoB_20_1", "AtoB_20_2", "AtoB_22.5_1", "AtoB_22.5_2", "AtoB_22.5_3",
         "AtoB_22.5_4", "BtoA_22.5_1", "BtoA_22.5_2", "BtoA_22.5_3", "BtoA_25_1",
         "BtoA_25_2", "BtoA_27.5_1", "BtoA_27.5_2", "BtoA_27.5_4"]
TEST = ["BtoA_27.5_3", "AtoB_20_4", "BtoA_25_3"]
VALIDATION = ["AtoB_20_3", "BtoA_25_4", "BtoA_22.5_4"]   # user's own; never loaded
assert len(set(TRAIN + TEST + VALIDATION)) == 20

def group_of(run_id: str) -> str:
    """'BtoA_27.5_3' -> 'BtoA_27.5' (direction + nominal speed)."""
    return run_id.rsplit("_", 1)[0]
