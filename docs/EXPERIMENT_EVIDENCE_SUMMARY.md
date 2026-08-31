# Experiment Evidence Summary / 实验与复验证据摘要

This document merges the historical experiment inventory with the multi-seed, regime-stratified, and cross-time-block audit notes. It is a concise evidence index, not a substitute for authorized raw data or model artifacts.

本文合并历史实验总览、多随机种子、工况分层与跨时间块复验要点。它是简明证据索引，不替代经授权的原始数据或模型产物。

## Scope and boundary / 范围与边界

- Results from different targets, horizons, physical conversions, and split protocols must not be compared directly.
- Most historical results are exploratory; they are not deployment claims.
- Reproduction requires controlled access to the original data and artifacts.

- 不同目标变量、预测时域、物理换算口径和数据切分协议的结果不可直接横向比较。
- 多数历史结果属于探索性证据，不构成部署结论。
- 完整复现需要受控访问原始数据和实验产物。

## Key historical findings / 历史关键发现

| Topic / 主题 | Observation / 观察结果 | Boundary / 边界 |
|---|---|---|
| Direct vs indirect soft sensing / 直接与间接软测量 | Direct V prediction outperformed M-to-V conversion in the cited h40 LSTM experiment. / 引用的 h40 LSTM 实验中，直接预测 V 优于先预测 M 再换算 V。 | Historical, protocol-specific. / 属于历史且特定协议结论。 |
| Forward prediction / 前视预测 | Deep models improved over last-reading in several 10–20 minute V forecasts. / 部分 10–20 分钟 V 前视任务中，深度模型优于 last-reading。 | Operating regime and split matter. / 受工况和切分影响。 |
| Regime behavior / 工况规律 | Linear models often performed strongly in ramp-down regimes; deep models could be competitive in steady regimes. / 线性模型在降负荷工况常有优势，深度模型在稳态可能更具竞争力。 | Not a universal deployment rule. / 不是通用部署规则。 |
| Window length / 窗口长度 | Window length is an independent experimental variable; longer windows can help some h80 linear baselines. / 窗口长度是独立实验变量，较长窗口可改善部分 h80 线性基线。 | Requires target-specific validation. / 需要针对目标重新验证。 |

## Multi-seed audit (BM-SEED-02) / 多 Seed 复验

Status: `AUDITED_EXPLORATORY`. Three seeds (7, 19, 42), horizons h40 and h80, window 20, and 15-second sampling were audited against manifests and per-sample metrics.

状态：`AUDITED_EXPLORATORY`。审计覆盖 seed 7、19、42，h40 和 h80，窗口 20、采样间隔 15 秒；逐样本指标与 manifest 已复算核对。

- Ridge achieved the lowest mean MAE for both horizons in this audit; Bayesian Ridge was close and stable.
- LSTM showed notable seed variance, so a single best run should not be treated as the multi-seed result.
- The batch can support historical retrieval and hypothesis generation, but needs further review before being elevated to confirmatory evidence.

- Ridge 在本审计的两个时域均取得最低平均 MAE；Bayesian Ridge 接近且稳定。
- LSTM 存在较明显的 seed 波动，不能用单次最优替代多 seed 结论。
- 本批结果可支持历史检索和假设生成，但升级为确认性证据前仍需进一步复核。

## Regime-stratified audit (BM-REGIME-01) / 工况分层审计

Status: `AUDITED_EXPLORATORY`. The audit reused BM-SEED-02 predictions without retraining models. Overall h40 favored Ridge, whereas rankings varied across ramp-up, ramp-down, and direction-change regimes. At h80, the winning model also varied substantially by regime and seed.

状态：`AUDITED_EXPLORATORY`。该审计复用 BM-SEED-02 预测结果，未重新训练模型。h40 整体由 Ridge 占优，但上升、下降与方向变化工况会发生排序翻转；h80 的最优模型也明显随工况和 seed 变化。

## Cross-time-block audit (BM-TIME-01) / 跨时间块复验

Status: `PHASE1_COMPLETED_EXPLORATORY`. Only Persistence, Ridge, and Bayesian Ridge were evaluated with seed 42. Results changed across early, middle, late, and latest-holdout blocks, showing that temporal drift must be considered before model selection.

状态：`PHASE1_COMPLETED_EXPLORATORY`。第一阶段仅评估 Persistence、Ridge 与 Bayesian Ridge，seed 固定为 42。不同时间块的最优模型发生变化，说明模型选择必须考虑时间漂移。

## Use in reports / 报告使用建议

Use these results with their exact target, horizon, split, and audit status. Do not convert exploratory findings into physical or deployment claims without an authorized replication and domain review.

引用这些结果时必须同时保留目标变量、预测时域、数据切分与审计状态；未经授权复现和领域审查，不得把探索性发现写成物理机理或部署结论。
