from pathlib import Path
import ast

path = Path(
    r"D:\BoilerMindTeamTest\_bm_sync_tmp\boilermind-research-v01\skills\industrial\steam_volume_model_benchmark\benchmark_pipeline.py"
)

text = path.read_text(
    encoding="utf-8-sig",
    errors="replace"
)

tree = ast.parse(text)

wanted = {
    "_engineer_common_features",
    "_mass_metrics",
    "_mass_segment_bundle",
}

for node in ast.walk(tree):
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        if node.name in wanted:
            print("\n" + "=" * 100)
            print(node.name)
            print("=" * 100)
            print(ast.get_source_segment(text, node))
