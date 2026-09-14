from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

from .config import DEFAULT_ARTIFACT_DIR, DEFAULT_DATA_DIR
from .inference import InferenceService

TOOLS = [
    {
        "name": "list_phm_models",
        "description": "List locally available FD001 RUL models and saved metrics.",
        "inputSchema": {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
    },
    {
        "name": "inspect_engine",
        "description": "Check one prepared FD001 test engine's final window.",
        "inputSchema": {
            "type": "object",
            "properties": {"unit_id": {"type": "integer", "minimum": 1}},
            "required": ["unit_id"],
            "additionalProperties": False,
        },
    },
    {
        "name": "predict_rul",
        "description": "Predict FD001 RUL with a saved LSTM or Transformer.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "unit_id": {"type": "integer", "minimum": 1},
                "model": {
                    "type": "string",
                    "enum": ["lstm", "transformer"],
                },
                "version": {"type": "string", "default": "active"},
            },
            "required": ["unit_id", "model"],
            "additionalProperties": False,
        },
    },
    {
        "name": "compare_rul_models",
        "description": "Run both FD001 models and select by validation RMSE.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "unit_id": {"type": "integer", "minimum": 1},
                "version": {"type": "string", "default": "active"},
            },
            "required": ["unit_id"],
            "additionalProperties": False,
        },
    },
    {
        "name": "get_model_metrics",
        "description": "Read saved PHM metrics without retraining a model.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "model": {
                    "type": "string",
                    "enum": ["lstm", "transformer"],
                },
                "version": {"type": "string", "default": "active"},
            },
            "required": ["model"],
            "additionalProperties": False,
        },
    },
    {
        "name": "get_degradation_evidence",
        "description": "Return strongest normalized trends in an engine window.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "unit_id": {"type": "integer", "minimum": 1},
                "top_k": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 10,
                    "default": 5,
                },
            },
            "required": ["unit_id"],
            "additionalProperties": False,
        },
    },
]


def _service() -> InferenceService:
    data_dir = Path(os.environ.get("PHM_DATA_DIR", str(DEFAULT_DATA_DIR))).resolve()
    artifact_dir = Path(os.environ.get("PHM_ARTIFACT_DIR", str(DEFAULT_ARTIFACT_DIR))).resolve()
    return InferenceService(data_dir, artifact_dir)


def _tool_result(payload: Any, is_error: bool = False) -> dict:
    return {
        "content": [{"type": "text", "text": json.dumps(payload, ensure_ascii=False)}],
        "isError": is_error,
    }


def call_tool(service: InferenceService, name: str, args: dict) -> dict:
    try:
        if name == "list_phm_models":
            return _tool_result({"models": service.list_models()})
        if name == "inspect_engine":
            return _tool_result(service.inspect_engine(int(args["unit_id"])))
        if name == "predict_rul":
            return _tool_result(
                service.predict_rul(
                    int(args["unit_id"]),
                    str(args["model"]),
                    str(args.get("version", "active")),
                )
            )
        if name == "compare_rul_models":
            return _tool_result(
                service.compare_models(int(args["unit_id"]), str(args.get("version", "active")))
            )
        if name == "get_model_metrics":
            model = str(args["model"])
            requested_version = str(args.get("version", "active"))
            version = service.registry.resolve(model, requested_version)
            matching = [
                item
                for item in service.list_models()
                if item["model"] == model and item["version"] == version
            ]
            if not matching:
                raise FileNotFoundError(f"Model artifact not found: {model}/{version}")
            return _tool_result(matching[0])
        if name == "get_degradation_evidence":
            return _tool_result(
                service.degradation_evidence(int(args["unit_id"]), int(args.get("top_k", 5)))
            )
        return _tool_result({"error": f"Unknown tool: {name}"}, True)
    except (KeyError, TypeError, ValueError, FileNotFoundError) as error:
        return _tool_result({"error": str(error)}, True)


def handle(request: dict, service: InferenceService) -> dict | None:
    method = request.get("method")
    request_id = request.get("id")
    if method == "notifications/initialized":
        return None
    if method == "initialize":
        result = {
            "protocolVersion": "2024-11-05",
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "mini-claude-phm", "version": "0.2.0"},
        }
    elif method == "tools/list":
        result = {"tools": TOOLS}
    elif method == "tools/call":
        params = request.get("params") or {}
        result = call_tool(
            service,
            str(params.get("name", "")),
            params.get("arguments") or {},
        )
    else:
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {"code": -32601, "message": f"Unknown method: {method}"},
        }
    return {"jsonrpc": "2.0", "id": request_id, "result": result}


def main() -> None:
    service = _service()
    for line in sys.stdin:
        try:
            response = handle(json.loads(line), service)
            if response is not None:
                print(json.dumps(response, ensure_ascii=False), flush=True)
        except (json.JSONDecodeError, TypeError):
            continue


if __name__ == "__main__":
    main()
