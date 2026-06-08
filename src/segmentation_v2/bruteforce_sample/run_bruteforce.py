from __future__ import annotations

import sys
from pathlib import Path


SUITE_DIR = Path(__file__).resolve().parent
SEGMENTATION_DIR = SUITE_DIR.parent
if str(SEGMENTATION_DIR) not in sys.path:
    sys.path.insert(0, str(SEGMENTATION_DIR))

from src.bruteforce_runner import main


if __name__ == "__main__":
    raise SystemExit(main(SUITE_DIR, expected_prefix="bf_sample_"))
