# Mini Claude PHM extension

A local, read-only MCP server for NASA C-MAPSS FD001 remaining useful life
prediction with a two-layer LSTM and a small Transformer encoder.

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

The downloader tries the NASA Open Data portal first. If that endpoint is
unavailable, it uses a commit-pinned read-only mirror and verifies every FD001
file against a repository-pinned SHA-256 digest.

Training is an explicit offline operation. The MCP server exposes inference,
saved metrics, data-quality checks, and descriptive sensor trends; it never
trains a model or modifies operational state during a chat.

## Agent integration

Merge `mcp.example.json` into the repository `.mcp.json`, then run `/doctor` or
`/mcp tools`. Invoke `/phm` with a request such as:

```text
/phm Compare the LSTM and Transformer RUL predictions for FD001 test engine 42.
```

## Limitations

- FD001 is simulated, single-condition, single-fault-mode data.
- The empirical interval is derived from validation residual quantiles, not a
  calibrated probabilistic model.
- Sensor trend evidence is descriptive. It is not a causal fault diagnosis and
  must not be used for real aircraft release-to-service decisions.
