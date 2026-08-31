from pathlib import Path

files = [
    Path(r"D:\BoilerMindTeamTest\_bm_sync_tmp\boilermind-research-v01\examples\golden_case_steam_soft_sensor\real_data\real_data_column_mapping.json"),
    Path(r"D:\BoilerMindTeamTest\_bm_sync_tmp\boilermind-research-v01\configs\authorized_boiler_181var_q006.json"),
    Path(r"D:\BoilerMindTeamTest\_bm_sync_tmp\boilermind-research-v01\configs\authorized_boiler_181var_rampdown.json"),
    Path(r"D:\BoilerMindTeamTest\_bm_sync_tmp\boilermind-research-v01\core\contracts\real_data_column_mapping_v11.py"),
]

keywords = [
    "feedwater", "fuel", "coal",
    "给水", "给煤", "煤量",
    "pressure", "temperature", "load",
    "mass_flow", "steam",
    "压力", "温度", "负荷", "蒸汽"
]

for path in files:
    print("\n" + "=" * 90)
    print(path)
    print("=" * 90)

    if not path.exists():
        print("NOT FOUND")
        continue

    text = path.read_text(encoding="utf-8-sig", errors="replace")

    found = False
    for i, line in enumerate(text.splitlines(), start=1):
        low = line.lower()
        if any(k.lower() in low for k in keywords):
            print(f"{i:04d}: {line}")
            found = True

    if not found:
        print("NO MATCH")

