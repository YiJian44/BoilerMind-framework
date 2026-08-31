#!/usr/bin/env bash
# 后台启 ResearchOrchestrator，轮询 run.json 看进度，最后退出
set -u
PROJECT="E:/AI-Workspace/30_Projects/active/BoilerMind 正式版有科研假设的更新迭代"
RUN_ID="${1:?run_id required}"
QUESTION="${2:?question required}"
TIMEOUT_MIN="${3:-15}"
PY="${PY:-/E/conda_envs/pytorch_env/python.exe}"
KEY=$(grep '^DASHSCOPE_API_KEY=' "${PROJECT}/.env.local" | head -1 | cut -d= -f2-)

export PYTHONUTF8=1 PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1
export BOILERMIND_QWEN_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
export BOILERMIND_QWEN_MODEL="qwen3.7-plus"
export BOILERMIND_QWEN_TIMEOUT=240 BOILERMIND_QWEN_MAX_TOKENS=4096
export BOILERMIND_ENABLE_WEB_LITERATURE=0
export DASHSCOPE_API_KEY="$KEY"
export BOILERMIND_REAL_DATASET_PATH="${PROJECT}/resources/data/shortperiod_new.csv"
export PYTHONPATH="${PROJECT}/src"

LOG="${PROJECT}/runtime/stdout_${RUN_ID}.log"
mkdir -p "${PROJECT}/runtime" "${PROJECT}/runtime/research_runs_v2"
cd "$PROJECT"

"$PY" -u scripts/run_full_e2e.py --question "$QUESTION" --run-id "$RUN_ID" >"$LOG" 2>&1 &
BGPID=$!
echo "[launch] pid=$BGPID run_id=$RUN_ID timeout=${TIMEOUT_MIN}min"
echo "[launch] log=$LOG"

START=$(date +%s)
LAST_SIZE=0
while true; do
  sleep 20
  if ! kill -0 "$BGPID" 2>/dev/null; then
    echo "[probe] process exited naturally"
    break
  fi
  ELAPSED=$(( $(date +%s) - START ))
  if [ "$ELAPSED" -ge $((TIMEOUT_MIN*60)) ]; then
    echo "[probe] timeout (${TIMEOUT_MIN}m) -- killing PID=$BGPID"
    kill -9 "$BGPID" 2>/dev/null
    break
  fi
  RUNF="${PROJECT}/runtime/research_runs_v2/${RUN_ID}/run.json"
  if [ -f "$RUNF" ]; then
    SIZE=$(stat -c%s "$RUNF" 2>/dev/null || stat -f%z "$RUNF")
    python - "$RUNF" "$ELAPSED" <<'PY'
import json,sys,os
p=sys.argv[1]; e=int(sys.argv[2])
try:
  d=json.load(open(p,encoding='utf-8'))
except Exception as ex:
  print(f"[probe t={e:>4}s] parse_error={ex}");sys.exit()
st=d.get('status','?')
stages=[s['stage'] for s in d.get('stage_traces',[])]
batches=[(b.get('batch_id'),b.get('status'),len(b.get('members',[]))) for b in d.get('batches',[])]
n_hyp=len(d.get('hypotheses',[]))
rankings=len(d.get('ranking_snapshots',[]))
errs=d.get('errors',[])
print(f"[probe t={e:>4}s] status={st} stages={stages} hypotheses={n_hyp} rankings={rankings} batches={batches} errors={len(errs)}")
for batch in d.get('batches',[]):
  for m in batch.get('members',[]):
    oid=(m.get('outcome') or {})
    er=oid.get('experiment_result',{}) if oid else {}
    met=er.get('metrics',{}) if er else {}
    sr=oid.get('scientific_result',{}) if oid else {}
    mae=met.get('mae_t_h')
    mae_s=f"{mae:.4f}" if isinstance(mae,(int,float)) else 'n/a'
    cands=list((er.get('candidate_locked_test_metrics') or {}).keys())
    print(f"  - {m.get('hypothesis_id','?'):>5} {m.get('status','?'):>9} exec={er.get('experiment_id','?')} verdict={sr.get('verdict','?')} MAE={mae_s} candidates={cands}")
PY
  else
    echo "[probe t=${ELAPSED}s] no run.json yet"
  fi
done
wait $BGPID 2>/dev/null
echo "[done] exit_code=$? log_tail:"
tail -40 "$LOG" 2>/dev/null
