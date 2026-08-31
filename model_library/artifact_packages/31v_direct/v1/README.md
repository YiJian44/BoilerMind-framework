# BoilerMind 31/V direct 模型库材料包（25146 样本）

31特征 → V(m³/s) 直接软测量；window=20；前视 h40(10min)/h80(20min)；70/10/20 时序切分；train-only 缩放；validation 选模；locked_test 评估。

## 内容
- `model_library/`：json(28条) + weights + predictions(逐样本) + manifests + logs + SHA256SUMS.json
- `environment/`：训练环境导出（python/pip_freeze/torch/hardware）
- `runtime/31v_data/`：窗口化缓存(npz) + feature scaler + dataset_manifest
- `data/boiler_181var_clean.csv`：原始数据（重建用）
- `scripts/`：5 个可复现脚本

## 关键身份
- dataset_sha256: `9c099b793c6d63edaeb6b3514415e5ba209eb2bf6ac5c940743485eebd56891c`
- 训练脚本 git commit: `ed08315`，scripts_source_sha256 见各 manifest

## 版本/兼容
- 训练环境：python3.10.12 / scikit-learn **1.7.2** / torch 2.7.1+cu128 / numpy 2.2.6 / pandas 2.3.3
- **sklearn joblib 权重版本锁定 1.7.2**：跨版本加载需匹配环境或重训；torch .pth 版本无关
- 逐样本预测 CSV 可直接复算 MAE/RMSE/R²/MBE（已抽查与登记值 diff=0）

## 审计说明
- 各模型日志含 `fit_converged`/`warnings`/`convergence_note`：为**推断值**（28/28 fit_success；elasticnet max_iter=5000、mlp early_stopping=True；warning 捕获未 instrument）。如需独立验证收敛状态可重跑。
- 特征名称：31 个特征仅 3 个有已核验名称（col1 压力 / col6 负荷 / col9 温度），其余以列号标识（变量映射仅核验了这 3 个）。
- 被排除模型：patchtst/itransformer/timesnet（用户判断垫底且一直 adapter_ready=False）、gpr（test MAE 3.82 越界，框架 RESOURCE_SAFETY 限制）——均未删除任何已训练产物，仅为未纳入。

## 重建
```bash
python scripts/build_31v_dataset.py --data data/boiler_181var_clean.csv --horizon 40 80 --out runtime/31v_data
python scripts/train_31v_library.py --data data/boiler_181var_clean.csv --cache runtime/31v_data --device cpu --max-epochs 100 --patience 15
```
