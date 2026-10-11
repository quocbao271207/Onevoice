import json
from pathlib import Path

p = Path("gpu-runs/model-bakeoff-20261005/bakeoff_state.json")
if p.exists():
    d = json.loads(p.read_text(encoding="utf-8"))
    comp_file = Path(d.get("selection_comparison", ""))
    if comp_file.exists():
        comp = json.loads(comp_file.read_text(encoding="utf-8"))
        print("Comparison keys:", list(comp.keys()))
        print("MT summary:", comp.get("mt", {}).get("winner"))
        print("ASR summary:", comp.get("asr", {}).get("winner"))
        print("Overall summary:", comp.get("summary"))
    else:
        print("Comparison file does not exist:", comp_file)
else:
    print("Not found")
