import hashlib
import io
import urllib.error
from pathlib import Path

import numpy as np
import phm_mcp.data as data_module
import pytest
from phm_mcp.data import download_fd001, split_engine_ids


@pytest.mark.parametrize("fraction", [0.0, 1.0, -0.1, 1.1])
def test_engine_split_rejects_invalid_fraction(fraction: float) -> None:
    rows = np.asarray([[unit_id, cycle, *([1.0] * 24)] for unit_id in (1, 2) for cycle in (1, 2)])
    with pytest.raises(ValueError, match="between 0 and 1"):
        split_engine_ids(rows, validation_fraction=fraction)


def test_download_uses_checksum_verified_mirror_on_official_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payloads = {
        "train_FD001.txt": b"train data\n",
        "test_FD001.txt": b"test data\n",
        "RUL_FD001.txt": b"rul data\n",
    }
    for name, payload in payloads.items():
        monkeypatch.setitem(data_module.FD001_SHA256, name, hashlib.sha256(payload).hexdigest())

    def fake_urlopen(url: str, timeout: int) -> io.BytesIO:
        assert timeout in (30, 120)
        if url == "https://official.invalid/archive.zip":
            raise urllib.error.URLError("offline")
        return io.BytesIO(payloads[url.rsplit("/", 1)[-1]])

    monkeypatch.setattr(data_module.urllib.request, "urlopen", fake_urlopen)
    with pytest.warns(RuntimeWarning, match="pinned FD001 mirror"):
        paths = download_fd001(
            tmp_path,
            url="https://official.invalid/archive.zip",
            mirror_url="https://mirror.invalid/fixed-commit",
        )

    assert [path.read_bytes() for path in paths] == list(payloads.values())
    assert not list(tmp_path.rglob("*.part"))
