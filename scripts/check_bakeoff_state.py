import json
from pathlib import Path

p = Path("gpu-runs/model-bakeoff-20261005/bakeoff_state.json")
if p.exists():
    d = json.loads(p.read_text(encoding="utf-8"))
    comp_file = Path(d.get("selection_comparison", ""))
    if comp_file.exists():
        comp = json.loads(comp_file.read_text(encoding="utf-8"))
        print("Status:", comp.get("status"))
        print("Selection blockers:", json.dumps(comp.get("selection_blockers"), indent=2))
        print("Results keys:", list(comp.get("results", {}).keys()))
    else:
        print("Comparison file does not exist:", comp_file)
else:
    print("Not found")
