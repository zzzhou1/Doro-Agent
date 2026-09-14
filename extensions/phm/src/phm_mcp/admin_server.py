from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

from .admin import AdminService
from .config import DEFAULT_ARTIFACT_DIR, DEFAULT_DATA_DIR, DEFAULT_RUNTIME_DIR

TOOLS = [
    {
        "name": "submit_training_job",
        "description": "Queue an asynchronous LSTM or Transformer training job.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "model": {"type": "string", "enum": ["lstm", "transformer"]},
                "version": {"type": "string", "minLength": 1, "maxLength": 64},
                "epochs": {"type": "integer", "minimum": 1, "maximum": 500, "default": 50},
                "batch_size": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 4096,
                    "default": 64,
                },
                "learning_rate": {
                    "type": "number",
                    "exclusiveMinimum": 0,
                    "maximum": 1,
                    "default": 0.001,
                },
                "patience": {"type": "integer", "minimum": 1, "default": 8},
                "seed": {"type": "integer", "minimum": 0, "default": 42},
                "device": {
                    "type": "string",
                    "pattern": "^(auto|cpu|cuda(:[0-9]+)?|mps)$",
                    "default": "auto",
                },
                "amp": {"type": "boolean", "default": False},
            },
            "required": ["model"],
            "additionalProperties": False,
        },
    },
    {
        "name": "get_training_job",
        "description": "Read status, progress, metrics, or errors for one training job.",
        "inputSchema": {
            "type": "object",
            "properties": {"job_id": {"type": "string"}},
            "required": ["job_id"],
            "additionalProperties": False,
        },
    },
    {
        "name": "list_training_jobs",
        "description": "List recent queued and completed PHM training jobs.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "limit": {"type": "integer", "minimum": 1, "maximum": 200, "default": 20}
            },
            "additionalProperties": False,
        },
    },
    {
        "name": "cancel_training_job",
        "description": "Cancel a queued job or request cooperative cancellation of a running job.",
        "inputSchema": {
            "type": "object",
            "properties": {"job_id": {"type": "string"}},
            "required": ["job_id"],
            "additionalProperties": False,
        },
    },
    {
        "name": "promote_candidate_model",
        "description": "Promote a succeeded candidate job to the active model registry.",
        "inputSchema": {
            "type": "object",
            "properties": {"job_id": {"type": "string"}},
            "required": ["job_id"],
            "additionalProperties": False,
        },
    },
    {
        "name": "rollback_phm_model",
        "description": "Roll one model family back to its previous active version.",
        "inputSchema": {
            "type": "object",
            "properties": {"model": {"type": "string", "enum": ["lstm", "transformer"]}},
            "required": ["model"],
            "additionalProperties": False,
        },
    },
    {
        "name": "get_training_control_status",
        "description": "Read the model registry and whether queued jobs require a worker.",
        "inputSchema": {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
    },
]


def _service() -> AdminService:
    data_dir = Path(os.environ.get("PHM_DATA_DIR", str(DEFAULT_DATA_DIR))).resolve()
    artifact_dir = Path(os.environ.get("PHM_ARTIFACT_DIR", str(DEFAULT_ARTIFACT_DIR))).resolve()
    runtime_dir = Path(os.environ.get("PHM_RUNTIME_DIR", str(DEFAULT_RUNTIME_DIR))).resolve()
    return AdminService(data_dir, artifact_dir, runtime_dir)


def _tool_result(payload: Any, is_error: bool = False) -> dict:
    return {
        "content": [{"type": "text", "text": json.dumps(payload, ensure_ascii=False)}],
        "isError": is_error,
    }


def call_tool(service: AdminService, name: str, args: dict) -> dict:
    try:
        if name == "submit_training_job":
            return _tool_result(service.submit_training_job(**args))
        if name == "get_training_job":
            return _tool_result(service.get_training_job(str(args["job_id"])))
        if name == "list_training_jobs":
            return _tool_result(service.list_training_jobs(int(args.get("limit", 20))))
        if name == "cancel_training_job":
            return _tool_result(service.cancel_training_job(str(args["job_id"])))
        if name == "promote_candidate_model":
            return _tool_result(service.promote_candidate_model(str(args["job_id"])))
        if name == "rollback_phm_model":
            return _tool_result(service.rollback_model(str(args["model"])))
        if name == "get_training_control_status":
            return _tool_result(
                {
                    "registry": service.get_model_registry(),
                    "worker": service.worker_status(),
                }
            )
        return _tool_result({"error": f"Unknown tool: {name}"}, True)
    except (KeyError, TypeError, ValueError, FileNotFoundError, FileExistsError) as error:
        return _tool_result({"error": str(error)}, True)


def handle(request: dict, service: AdminService) -> dict | None:
    method = request.get("method")
    request_id = request.get("id")
    if method == "notifications/initialized":
        return None
    if method == "initialize":
        result = {
            "protocolVersion": "2024-11-05",
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "mini-claude-phm-admin", "version": "0.2.0"},
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
