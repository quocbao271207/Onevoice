import json
from pathlib import Path

p = Path("/workspace/test/Onevoice/gpu-runs/model-bakeoff-20261005/rescored_comparison.json")
if not p.exists():
    print("File not found")
    exit(1)

data = json.loads(p.read_text(encoding="utf-8"))

print("=" * 60)
print("ASR RESCORED RESULTS (BEFORE vs AFTER)")
print("=" * 60)
for name, res in data.get("asr", {}).items():
    print(f"\nModel: {name} (WER={res.get('wer', 0):.4f})")
    for cat, cdata in res.get("category_comparison", {}).items():
        old_f = cdata.get("old_failures")
        new_f = cdata.get("new_failures")
        old_r = cdata.get("old_failure_rate")
        new_r = cdata.get("new_failure_rate")
        print(f"  {cat:15s}: failures {old_f} -> {new_f} | rate {old_r:.3f} -> {new_r:.3f}")

print("\n" + "=" * 60)
print("MT RESCORED RESULTS (BEFORE vs AFTER)")
print("=" * 60)
for name, res in data.get("mt", {}).items():
    old_tot = res.get("old_safety_failure_rate")
    new_tot = res.get("new_safety_failure_rate")
    print(f"\nModel: {name}")
    print(f"  Overall Safety Failure Rate: {old_tot:.3f} -> {new_tot:.3f} (BLEU={res.get('sacrebleu', 0):.2f})")
    for cat, cdata in res.get("category_comparison", {}).items():
        old_r = cdata.get("old_failure_rate")
        new_r = cdata.get("new_failure_rate")
        old_iss = cdata.get("old_issues", {})
        new_iss = cdata.get("new_issues", {})
        print(f"  {cat:15s}: rate {old_r:.3f} -> {new_r:.3f} | issues {old_iss} -> {new_iss}")
