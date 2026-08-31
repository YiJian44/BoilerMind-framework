# BoilerMind 队友实验任务包

版本：`BM-RUNNER-0.2.0`

本包当前只授权环境检查和两项冒烟实验。`BM-SEED-02` 仍处于 `FORMAL_HOLD`，负责人确认冒烟结果后才会另发正式任务文件。

## 你需要完成什么

1. 将 ZIP 解压到一个全新目录，路径可以包含中文和空格。
2. 安装 Python 3.11 x64。
3. 在 PowerShell 中进入解压后的项目根目录。
4. 创建独立虚拟环境并安装依赖。
5. 运行环境检查，确认数据哈希一致。
6. 运行 Ridge 冒烟和 DLinear 冒烟。
7. 运行逐样本预测验收。
8. 将整个 `outputs/teammate` 目录压缩后回传给负责人。

## 第一步：配置环境

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\teammate\setup-environment.ps1
```

默认会安装 CPU/GPU 环境可用的 Torch。如果已经自行安装了正确的 Torch，可以使用：

```powershell
.\teammate\setup-environment.ps1 -SkipTorch
```

不要使用已有的全局 Python 环境，也不要把 `.venv` 回传。

## 第二步：检查环境与数据

将代号替换成负责人分配的代号，不要填写真实姓名或敏感电脑名称：

```powershell
.\teammate\check-environment.ps1 -OperatorId TEAMMATE-A -MachineId PC-A
```

必须看到：

```text
"status": "PASSED"
```

固定数据 SHA-256：

```text
d52c1399b844165f94fc156fc7919be9fbb0bf214dfff74b5c48bf429917759e
```

如果环境检查失败，停止，不要修改数据、任务文件或源码。

## 第三步：运行两项冒烟

```powershell
.\teammate\run-smoke.ps1 -TaskId BM-SMOKE-02-RIDGE-S42
.\teammate\run-smoke.ps1 -TaskId BM-SMOKE-02-DLINEAR-S42
```

每项任务都在独立子进程运行，并由父进程执行硬超时。超时属于工程失败，不是模型证伪。

## 第四步：复算预测指标

```powershell
.\teammate\verify-smoke.ps1 -Model ridge
.\teammate\verify-smoke.ps1 -Model dlinear
```

两项都必须看到：

```text
"passed": true
```

## 禁止事项

- 不得运行 `formal_tasks_HOLD.json`。
- 不得修改 seed、模型、epoch、window、horizon 或数据切分。
- 不得使用 locked-test 选择参数或模型。
- 不得用 Excel 打开并覆盖保存 CSV 数据。
- 不得手工修改 `result.json`、预测 CSV 或 manifest。
- 不得把 API key、`.env`、个人路径或 `.venv` 回传。
- 冒烟成功只代表执行链可复现，不代表形成科学结论。

## 回传内容

只需压缩并回传：

```text
outputs/teammate/
├── environment.json
├── model_runs/
│   ├── BM-SMOKE-02-RIDGE-S42/
│   └── BM-SMOKE-02-DLINEAR-S42/
└── supervisor/
    ├── BM-SMOKE-02-RIDGE-S42.json
    └── BM-SMOKE-02-DLINEAR-S42.json
```

模型文件较大时不要删除。若传输工具限制大小，先告诉负责人，由负责人决定是否仅回传模型哈希。

## 预期验收边界

- Ridge 应与负责人结果高度一致，锁定测试 MAE 约为 `27.937713 t/h`。
- DLinear 是 Torch 数值链路冒烟，3 epochs 下不要求优于 persistence。
- DLinear 预测必须已经回到真实 `t/h` 量纲，不能集中在 0 到 1。
- 两台机器 Torch 数值允许存在小幅差异，但样本数、索引、数据哈希和协议哈希必须一致。
