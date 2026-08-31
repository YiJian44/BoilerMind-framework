#!/usr/bin/env bash
# Run the 31/V direct model-library training on the Aliyun T4 server.
# Expects: scripts/ + boiler_181var_clean.csv in this dir (uploaded together).
set -euo pipefail
cd "$(dirname "$0")"
ROOT="$(cd .. && pwd)"

DATA="${1:-$ROOT/boiler_181var_clean.csv}"
CACHE="$ROOT/runtime/31v_data"
OUT_JSON="$ROOT/model_library/model_library.json"

echo "== data: $DATA =="
ls -lh "$DATA"

python3 build_31v_dataset.py --data "$DATA" --horizon 40 80 --out "$CACHE"

echo "== training (device=cuda, horizons 40 80, all 14 models) =="
python3 train_31v_library.py \
  --data "$DATA" \
  --cache "$CACHE" \
  --device cuda \
  --max-epochs 100 \
  --patience 15 \
  --out-json "$OUT_JSON"

echo "== artifacts =="
find "$ROOT/model_library" -type f | sort | head -60
echo
echo "DONE -> $OUT_JSON"
