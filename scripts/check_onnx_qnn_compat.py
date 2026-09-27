"""Fail-fast checks for ONNX patterns known to block QNN static compilation."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


def find_dynamic_shape_risks(
    nodes: Iterable[Any],
    initializer_names: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Return deterministic evidence for data-dependent shape operations."""
    node_list = list(nodes)
    producers = {output: node for node in node_list for output in node.output}
    initializers = initializer_names or set()
    risks: list[dict[str, Any]] = []
    for node in node_list:
        if node.op_type == "NonZero":
            risks.append(
                {
                    "node": node.name or "<unnamed>",
                    "op_type": node.op_type,
                    "severity": "block",
                    "reason": "NonZero produces a data-dependent output shape",
                }
            )
        if node.op_type != "Range" or len(node.input) < 2:
            continue
        limit = node.input[1]
        producer = producers.get(limit)
        if limit in initializers or producer is None or producer.op_type == "Constant":
            continue
        severity = "block" if producer.op_type == "ReduceMax" else "warning"
        risks.append(
            {
                "node": node.name or "<unnamed>",
                "op_type": node.op_type,
                "severity": severity,
                "reason": "Range limit is computed at runtime",
                "limit_tensor": limit,
                "limit_producer": producer.op_type,
                "limit_producer_node": producer.name or "<unnamed>",
            }
        )
    return sorted(
        risks,
        key=lambda item: (item["node"], item["op_type"], item["severity"]),
    )


def analyze_model(path: str | Path) -> dict[str, Any]:
    try:
        import onnx
    except ImportError as exc:
        raise RuntimeError(
            "onnx is required; run this check with .venv-aihub-models"
        ) from exc

    source = Path(path)
    model = onnx.load(str(source), load_external_data=False)
    nodes = list(model.graph.node)
    risks = find_dynamic_shape_risks(
        nodes,
        {initializer.name for initializer in model.graph.initializer},
    )
    blocking_risks = [risk for risk in risks if risk["severity"] == "block"]
    return {
        "model": str(source),
        "node_count": len(nodes),
        "op_counts": dict(sorted(Counter(node.op_type for node in nodes).items())),
        "dynamic_shape_risks": risks,
        "blocking_risk_count": len(blocking_risks),
        "qnn_static_shape_gate": "fail" if blocking_risks else "pass",
    }


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("models", nargs="+")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    payload = {"models": [analyze_model(path) for path in args.models]}
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 1 if any(
        model["qnn_static_shape_gate"] == "fail" for model in payload["models"]
    ) else 0


if __name__ == "__main__":
    raise SystemExit(main())
