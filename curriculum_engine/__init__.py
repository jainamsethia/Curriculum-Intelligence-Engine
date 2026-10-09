"""Curriculum Intelligence Engine (Group 9): overlap, redundancy and missing-prerequisite analysis of NMIMS syllabi."""
import os
from pathlib import Path

__version__ = "0.2.0"
ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("CE_DATA_DIR", ROOT / "data"))          # NMIMS-derived data, never committed
RESULTS = ROOT / "results"
LABELS = ROOT / "labels"
ZIP = Path(os.environ.get("NMIMS_ZIP", ROOT / "B TECH.zip"))
SEED = 42
