# Mini Claude PHM extension

A local NASA C-MAPSS FD001 remaining-useful-life system with a two-layer LSTM
and a small Transformer encoder. It exposes a read-only inference MCP server
and a separate write-capable training-control MCP server backed by SQLite and
a background worker.

## Reproduce

```powershell
uv sync --project extensions/phm --extra test
uv run --project extensions/phm phm download
uv run --project extensions/phm phm prepare
uv run --project extensions/phm phm train --model lstm
uv run --project extensions/phm phm train --model transformer
uv run --project extensions/phm phm evaluate
uv run --project extensions/phm phm predict --unit-id 42
```

## Asynchronous training

```powershell
# Terminal 1: long-running single worker
uv run --project extensions/phm phm-worker

# Terminal 2: queue and inspect a candidate
uv run --project extensions/phm phm-admin submit --model lstm --version v2 --epochs 20 --patience 5 --device auto
uv run --project extensions/phm phm-admin list
uv run --project extensions/phm phm-admin status JOB_ID
uv run --project extensions/phm phm-admin promote JOB_ID
uv run --project extensions/phm phm-admin rollback --model lstm
```

The downloader tries the NASA Open Data portal first. If that endpoint is
unavailable, it uses a commit-pinned read-only mirror and verifies every FD001
file against a repository-pinned SHA-256 digest.

Training jobs use the already prepared static FD001 dataset; there is no live
sensor ingestion. Submission returns immediately, while `phm-worker` trains on
CPU, CUDA, CUDA device index, or Apple MPS according to `--device`. A completed
job remains a candidate until explicitly promoted. Promotion atomically updates
the active-version registry, and rollback restores the previous active version.
Inference can continue while the worker trains.

## Agent integration

Merge both entries from `mcp.example.json` into the repository `.mcp.json`, then
run `/doctor` or `/mcp tools`. `phm` must stay `readOnly: true`; `phm-admin` must
stay `readOnly: false` because it queues jobs and changes the active registry.
Invoke `/phm` with a request such as:

```text
/phm Compare the LSTM and Transformer RUL predictions for FD001 test engine 42.
```

## Limitations

- FD001 is simulated, single-condition, single-fault-mode data.
- The empirical interval is derived from validation residual quantiles, not a
  calibrated probabilistic model.
- Sensor trend evidence is descriptive. It is not a causal fault diagnosis and
  must not be used for real aircraft release-to-service decisions.
- Online control means on-demand job submission and immediate post-promotion
  inference, not continuous learning or live telemetry ingestion.
