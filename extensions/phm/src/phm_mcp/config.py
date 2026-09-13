from __future__ import annotations

from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = PACKAGE_ROOT / "data"
DEFAULT_ARTIFACT_DIR = PACKAGE_ROOT / "artifacts"
NASA_CMAPSS_URL = "https://data.nasa.gov/docs/legacy/CMAPSSData.zip"
FD001_MIRROR_COMMIT = "97cf10d200d07c6e9e20e75c52639ce6a08736ce"
FD001_MIRROR_URL = (
    "https://huggingface.co/datasets/DeveloperMindset123/"
    f"CMAPSS_Jet_Engine_Simulated_Data/resolve/{FD001_MIRROR_COMMIT}"
)

FD001_FILES = ("train_FD001.txt", "test_FD001.txt", "RUL_FD001.txt")
FD001_SHA256 = {
    "train_FD001.txt": "963b5e22825b34d8b21c69e1aeb4af3e647050eb672ee8834ba4b5d91d2de0f8",
    "test_FD001.txt": "3cda7109ce17bafb5443f2ac926cfcf88154b941b8c4cf95eb55d1ddd6f52851",
    "RUL_FD001.txt": "a19c8ec94931949d0485bdc35118206e9c81c4547b422efb9cf86f4ceddbceca",
}
WINDOW_SIZE = 30
RUL_CAP = 125.0
