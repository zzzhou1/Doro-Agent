# NASA C-MAPSS data

The raw data is intentionally not committed. Download the official NASA PCoE
archive and prepare FD001 with:

```powershell
uv run --project extensions/phm phm download
uv run --project extensions/phm phm prepare
```

Expected raw files:

- `raw/train_FD001.txt`
- `raw/test_FD001.txt`
- `raw/RUL_FD001.txt`

The downloader prefers NASA's current Open Data resource. Because that endpoint
can be intermittently unavailable, it falls back to a commit-pinned mirror and
accepts files only when their SHA-256 digests match the pinned FD001 manifest.

Source: NASA Prognostics Center of Excellence, "Turbofan Engine Degradation
Simulation Data Set" (Saxena and Goebel, 2008).
