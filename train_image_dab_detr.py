"""Train image dab_detr, then evaluate on the test split."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common.runner import main
from detectors.dab_detr.build import build_model

if __name__ == "__main__":
    main("train", task="image", version="dab_detr", build_model=build_model)
