from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
SERIES_DIR = DATA_DIR / "series"
TRAIN_CSV = DATA_DIR / "train.csv"
