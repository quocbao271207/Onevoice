import json
from pathlib import Path

p = Path("gpu-runs/model-bakeoff-20261005/bakeoff_state.json")
if p.exists():
    d = json.loads(p.read_text(encoding="utf-8"))
    print("workflow_stage:", d.get("workflow_stage"))
    print("status:", d.get("status"))
    print("comparison:", d.get("selection_comparison"))
    print("selection_winners:", d.get("selection_winners"))
    print("candidate_a:", d.get("candidate_a_frozen"))
else:
    print("Not found")
