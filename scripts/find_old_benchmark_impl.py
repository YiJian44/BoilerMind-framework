from pathlib import Path

root = Path(r"D:\BoilerMindTeamTest\_bm_sync_tmp")

markers = [
    "benchmark_feature_schema",
    "validation_mae_m3_s",
    "candidate_mae_improvement_ratio",
    "ahead_steam_volume_forecast",
    "main_steam_mass_flow",
    "Ridge(",
    "joblib.dump",
    "torch.save",
    "IAPWS",
    "specific_volume",
    "steam_volumetric_flow",
]

hits = []

for path in root.rglob("*.py"):
    if ".venv" in path.parts or "__pycache__" in path.parts:
        continue

    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        continue

    lines = text.splitlines()

    matched = []
    for i, line in enumerate(lines, start=1):
        if any(marker.lower() in line.lower() for marker in markers):
            matched.append((i, line.strip()))

    if matched:
        hits.append((path, matched))

out = Path(r"D:\BoilerMind-Trusted\runtime\old_benchmark_impl_hits.txt")

with out.open("w", encoding="utf-8") as f:
    for path, matched in hits:
        f.write("\n" + "=" * 100 + "\n")
        f.write(str(path) + "\n")
        f.write("=" * 100 + "\n")

        for i, line in matched[:40]:
            f.write(f"{i:05d}: {line}\n")

print("MATCHED_FILES =", len(hits))
print("SAVED =", out)

for path, matched in hits[:30]:
    print()
    print(path)
    for i, line in matched[:12]:
        print(f"  {i:05d}: {line}")
