import json
from pathlib import Path

p = Path("/workspace/test/Onevoice/gpu-runs/model-bakeoff-20261005/rescored_comparison.json")
data = json.loads(p.read_text(encoding="utf-8"))["asr"]
print(json.dumps(data, indent=2, ensure_ascii=False))
