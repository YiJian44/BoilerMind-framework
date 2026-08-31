# BM-SEED-02：队友多 Seed 复验教程

## 1. 本轮目的

在完全相同的数据、特征、时间切分和训练协议下，分别使用 seed `7`、`19`、`42` 重新训练代表模型，判断：

- 模型指标是否存在明显随机波动；
- 模型排名是否因 seed 改变而翻转；
- Ridge 与 Bayesian Ridge 是否持续近似等价；
- h40 的 Transformer 和 h80 的 LSTM 优势是否稳定；
- RF 的高分是否能在多个 seed 下保持。

本轮仍属于多 seed 复验，不是跨时间块复验，也不能单独证明模型具有最终部署价值。

## 2. 固定实验口径

| 字段 | 固定值 |
|---|---|
| 任务编号 | `BM-SEED-02` |
| 数据 | `resources/datasets/boiler_181var_v1/boiler_181var_clean.csv` |
| 数据 SHA-256 | `9c099b793c6d63edaeb6b3514415e5ba209eb2bf6ac5c940743485eebd56891c` |
| 输入 | 31 个软测量特征 |
| 目标 | 蒸汽体积流量 `V (m³/s)` |
| 历史窗口 | 20 步 |
| 预测跨度 | h40、h80 |
| 时间切分 | 70% train / 10% validation / 20% locked test |
| 缩放 | 只在 train 上拟合 |
| 选模 | 只允许使用 validation |
| locked test | 冻结模型后评价，不参与选模 |
| seeds | `7`、`19`、`42` |
| 模型 | Persistence、Ridge、Bayesian Ridge、RF、Transformer、LSTM |
| Torch 训练 | 最多 100 epochs，patience 15 |

选择 Transformer 和 LSTM 两个 Torch 模型，是因为现有 seed=42 结果中，h40 最佳 Torch 为 Transformer，h80 最佳 Torch 为 LSTM。两个模型在两个 horizon 上都运行，避免按结果事后删选。

## 3. 获取正确仓库版本

在 PowerShell 中进入你自己的 BoilerMind 仓库：

```powershell
Set-Location "你的BoilerMind仓库路径"
git rev-parse --show-toplevel
git status --short
git fetch origin
git switch codex/experiment-memory-hypothesis-loop
git pull --ff-only origin codex/experiment-memory-hypothesis-loop
git log -1 --oneline
```

如果 `git status --short` 显示你有未提交修改，先停止，不要强行切换或覆盖，截图发回。

## 4. 创建独立 Python 环境

建议使用 Python 3.10 或 3.11。不要使用系统中混有其他项目依赖的环境。

```powershell
python -m venv .venv-bm-seed02
Set-ExecutionPolicy -Scope Process Bypass
& ".\.venv-bm-seed02\Scripts\Activate.ps1"
python -m pip install --upgrade pip
python -m pip install -r requirements-server.txt
```

记录环境：

```powershell
New-Item -ItemType Directory -Force -Path "outputs\experiments\BM-SEED-02\environment" | Out-Null
python scripts\dump_env.py --out "outputs\experiments\BM-SEED-02\environment" --seed 42
```

检查 Torch 与 GPU：

```powershell
python -c "import torch; print('torch=',torch.__version__); print('cuda=',torch.cuda.is_available()); print('gpu=',torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"
```

有 NVIDIA GPU 且 `cuda=True` 时使用 `cuda`。否则把下方命令中的 `$device = "cuda"` 改为 `$device = "cpu"`，但运行会明显更慢。

## 5. 核对原始数据

```powershell
(Get-FileHash -LiteralPath "resources\datasets\boiler_181var_v1\boiler_181var_clean.csv" -Algorithm SHA256).Hash.ToLowerInvariant()
```

必须输出：

```text
9c099b793c6d63edaeb6b3514415e5ba209eb2bf6ac5c940743485eebd56891c
```

不一致时立即停止，不能继续跑。

## 6. 构建统一数据缓存

只需构建一次，三个 seed 共用同一份缓存：

```powershell
python scripts\build_31v_dataset.py `
  --data "resources\datasets\boiler_181var_v1\boiler_181var_clean.csv" `
  --horizon 40 80 `
  --out "runtime\31v_data"
```

确认以下文件存在：

```powershell
Get-ChildItem -LiteralPath "runtime\31v_data"
```

至少应看到 `h40.npz`、`h80.npz`、两个 scaler 文件和元数据文件。该目录已被 Git 忽略，不要手工加入 Git。

## 7. 执行三个 seed

先确认修复版参数存在：

```powershell
python scripts\train_31v_library.py --help | Select-String "sklearn-n-jobs|parallel-execution"
```

本轮拆成两部分：五个非 RF 模型按 seed 顺序运行；三个 RF seed 各限制为 2 个 CPU worker 后并行运行。并行 RF 的运行时间受资源争抢影响，只能比较预测指标，不能用于模型速度比较。

### 7.1 顺序运行非 RF 模型

```powershell
$device = "cuda"
$seeds = @(7, 19, 42)
$models = @("persistence", "ridge", "bayesianridge", "transformer", "lstm")
$experimentRoot = "outputs\experiments\BM-SEED-02"

foreach ($seed in $seeds) {
  $seedRoot = Join-Path $experimentRoot "seed_$seed\core"
  New-Item -ItemType Directory -Force -Path $seedRoot | Out-Null

  python scripts\train_31v_library.py `
    --horizon 40 80 `
    --models $models `
    --cache "runtime\31v_data" `
    --device $device `
    --max-epochs 100 `
    --patience 15 `
    --seed $seed `
    --out-root (Join-Path $seedRoot "model_library") `
    --out-json (Join-Path $seedRoot "model_library\model_library.json") `
    2>&1 | Tee-Object -FilePath (Join-Path $seedRoot "console.log")

  if ($LASTEXITCODE -ne 0) {
    Write-Error "BM-SEED-02 seed=$seed 运行失败，停止后续 seed。不要删除失败目录。"
    break
  }
}
```

### 7.2 并行运行三个 RF seed

8 核服务器上每个 RF 固定 `n_jobs=2`，三个进程最多使用约 6 个 worker。输出写入各 seed 的 `rf_corrected` 子目录，不会覆盖其他结果。

```powershell
$repoRoot = (Get-Location).Path
$experimentRoot = "outputs\experiments\BM-SEED-02"
$jobs = foreach ($seed in @(7, 19, 42)) {
  $rfRoot = Join-Path $experimentRoot "seed_$seed\rf_corrected"
  New-Item -ItemType Directory -Force -Path $rfRoot | Out-Null
  Start-Job -ArgumentList $repoRoot, $seed, $rfRoot -ScriptBlock {
    param($repoRoot, $seed, $rfRoot)
    Set-Location $repoRoot
    python scripts\train_31v_library.py `
      --horizon 40 80 `
      --models rf `
      --cache "runtime\31v_data" `
      --device cpu `
      --seed $seed `
      --sklearn-n-jobs 2 `
      --parallel-execution `
      --out-root (Join-Path $rfRoot "model_library") `
      --out-json (Join-Path $rfRoot "model_library\model_library.json") `
      2>&1 | Tee-Object -FilePath (Join-Path $rfRoot "console.log")
    if ($LASTEXITCODE -ne 0) { throw "RF seed=$seed failed with exit code $LASTEXITCODE" }
  }
}

$jobs | Wait-Job | Receive-Job
$jobs | Select-Object Id,State,HasMoreData
```

注意：

- Transformer/LSTM 不并行；只有固定为 `n_jobs=2` 的三个 RF 进程可以并行。
- RF 命令必须包含 `--parallel-execution`，让 manifest 明确记录共享主机资源。
- 不要修改模型集合、epochs、patience、horizon 或切分。
- 某模型失败时不得用其他模型替代，也不得删除失败日志。
- locked-test 指标再好，也不能反过来调整参数。

## 8. 完成后检查

```powershell
$experimentRoot = "outputs\experiments\BM-SEED-02"
Get-ChildItem -LiteralPath $experimentRoot -Recurse -File | Group-Object Extension | Select-Object Name,Count

foreach ($seed in @(7, 19, 42)) {
  foreach ($part in @("core", "rf_corrected")) {
    $jsonPath = Join-Path $experimentRoot "seed_$seed\$part\model_library\model_library.json"
    python -c "import json,sys; p=json.load(open(sys.argv[1],encoding='utf-8')); print(sys.argv[1], 'count=',p['count'])" $jsonPath
  }
}
```

每个 seed 的 `core/model_library.json` 应有 10 条成功记录（5 个模型 × 2 个 horizon），`rf_corrected/model_library.json` 应有 2 条成功记录（RF × 2 个 horizon）。如果数量不足，保留现场并报告失败模型，不要重跑覆盖。

每个 seed 目录至少应包含：

```text
seed_7/
  core/
    console.log
    model_library/...
  rf_corrected/
    console.log
    model_library/...
```

seed 19、42 的结构相同。

## 9. 打包回传

先生成整个实验目录的文件哈希：

```powershell
$experimentRoot = Resolve-Path "outputs\experiments\BM-SEED-02"
$hashRows = Get-ChildItem -LiteralPath $experimentRoot -Recurse -File | ForEach-Object {
  [pscustomobject]@{
    path = $_.FullName.Substring($experimentRoot.Path.Length + 1)
    size_bytes = $_.Length
    sha256 = (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
  }
}
$hashRows | ConvertTo-Json -Depth 3 | Set-Content -LiteralPath (Join-Path $experimentRoot "SHA256SUMS.json") -Encoding UTF8
```

再打包：

```powershell
$zip = "outputs\delivery\BM-SEED-02-results.zip"
New-Item -ItemType Directory -Force -Path "outputs\delivery" | Out-Null
Compress-Archive -LiteralPath "outputs\experiments\BM-SEED-02" -DestinationPath $zip -Force
Get-Item -LiteralPath $zip | Select-Object FullName,Length
Get-FileHash -LiteralPath $zip -Algorithm SHA256
git rev-parse HEAD
```

需要回传：

1. `BM-SEED-02-results.zip`
2. ZIP 文件大小
3. ZIP SHA-256
4. `git rev-parse HEAD` 输出的提交号
5. 使用的设备：CPU 或 GPU，以及 GPU 名称
6. 是否出现失败、warning、显存不足或中断

不要把 `outputs/experiments/BM-SEED-02`、权重或预测文件提交到 GitHub；只通过 ZIP 回传。

## 10. 结果解释边界

队友只负责真实执行和完整回传，不要人工挑选“最好结果”。收到三个 seed 后，由主仓库统一计算：

- 每个模型、每个 horizon 的均值和标准差；
- 相对 Persistence 的改善率；
- 三个 seed 中的胜率；
- validation 与 locked-test 排名翻转；
- 是否需要扩展到 seed `5`、`23`、`11`。

只有出现明显随机波动、模型排名翻转或结论不稳定时，才扩展到第二组三个 seed。
