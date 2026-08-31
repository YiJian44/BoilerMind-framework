# Unity WebGL Interface Report

审查来源（只读）：

- `D:\BoilerMind-Unity\UnityWebgl\unity-bridge.js`
- `D:\BoilerMind-Unity\UnityWebgl\server\state_server.py`
- `D:\BoilerMind-Unity\UnityWebgl\后端通讯手册.md`
- `D:\BoilerMind-Unity\UnityWebgl\test_panel.html`

当前部署目录未提供可读的 `*.cs` 或 `*.jslib`。因此下列契约以 JavaScript
实际路由、Python 实际广播结构、通讯手册及测试面板四者的交集为准；无法从
部署包独立验证 Unity C# 内部 DTO。

## 1. Unity 支持的消息类型

| `type` | Unity 方法 | 用途 |
|---|---|---|
| `connected` | JS 内部处理 | 连接确认及初始状态 |
| `chartData` | `WaterWallBridge.ReceiveChartData` | 实际/预测曲线 |
| `gaugeData` | `WaterWallBridge.ReceiveGaugeData` | 仪表数据 |
| `fault` | `WaterWallBridge.ReceiveFaultData` | 故障显示 |
| `thermal` | `WaterWallBridge.ReceiveThermalData` | 热力图 |
| `pipeThermal` | `WaterWallBridge.ReceivePipeThermalData` | 管道热力图 |
| `simResult` | `WaterWallBridge.ReceiveSimResult` | 锅炉模拟结果 |
| `targetResult` | `WaterWallBridge.ReceiveTargetResult` | 目标蒸汽量及推荐参数 |
| `question` | `WaterWallBridge.ReceiveQuestion` | 通知/实验摘要弹窗 |

无 `type` 且包含 `state` 的消息交给 `ReceiveMessage`。

## 2. `targetResult` 字段

基础字段是当前 bridge 日志和后端正式广播均直接读取/生成的字段，适配器只有在
四项全部来自 OptimizationResult 时才生成消息。

| 字段 | 类型 | 必填 | 含义 |
|---|---|---|---|
| `type` | string | 是 | 固定为 `targetResult` |
| `target_steam` | number | 是 | 目标蒸汽量，t/h |
| `wall_temp` | number | 是 | 管壁温度，°C |
| `state_code` | integer | 是 | 0正常、1过热、2泄漏、3爆管 |
| `state_name` | string | 是 | 状态名称 |
| `parameters` | object | 否 | 原始工况参数 |
| `rec_coal_feed` | number | 否 | 推荐给煤量，t/h |
| `rec_air_flow` | number | 否 | 推荐送风量，km³/h |
| `rec_water_flow` | number | 否 | 推荐给水流量，t/h |
| `rec_drum_pressure` | number | 否 | 推荐汽包压力，MPa |
| `rec_slag_degree` | number | 否 | 推荐结渣程度 |
| `rec_notes` | string | 否 | 第0组方案说明 |
| `rec1_*` … `rec4_*` | 同第0组 | 否 | 第1至4组推荐及说明 |
| `source` | string | 否 | 来源标识 |
| `timestamp` | string | 否 | ISO-8601 时间 |

### Python 输出模板

```python
{
    "type": "targetResult",
    "target_steam": optimization_result.target_steam,
    "wall_temp": optimization_result.wall_temp,
    "state_code": optimization_result.state_code,
    "state_name": optimization_result.state_name,
    # 以下字段仅在结果真实提供时输出
    "rec_coal_feed": optimization_result.rec_coal_feed,
    "rec_air_flow": optimization_result.rec_air_flow,
    "rec_water_flow": optimization_result.rec_water_flow,
    "rec_drum_pressure": optimization_result.rec_drum_pressure,
    "rec_slag_degree": optimization_result.rec_slag_degree,
    "rec_notes": optimization_result.rec_notes,
}
```

## 3. `chartData` 字段

| 字段 | 类型 | 必填 | 含义 |
|---|---|---|---|
| `type` | string | 是 | 固定为 `chartData` |
| `chartType` | string | 是 | 曲线类型 |
| `xLabels` | string[] | 是 | X轴标签 |
| `actualValues` | number[] | 是 | 实际值 |
| `predictedValues` | number[] | 是 | 预测值 |
| `source` | string | 否 | 来源 |
| `timestamp` | string | 否 | ISO-8601 时间 |

```python
{
    "type": "chartData",
    "chartType": curve["chartType"],
    "xLabels": curve["xLabels"],
    "actualValues": curve["actualValues"],
    "predictedValues": curve["predictedValues"],
}
```

## 4. `question` 字段

| 字段 | 类型 | 必填 | 含义 |
|---|---|---|---|
| `type` | string | 是 | 固定为 `question` |
| `title` | string | 是 | 标题 |
| `content` | string | 是 | 正文 |
| `severity` | string | 是 | `info` / `warning` / `critical` |
| `source` | string | 否 | 来源 |
| `timestamp` | string | 否 | ISO-8601 时间 |

`question` 与 `targetResult` 是不同语义通道。实验摘要不应伪装成控制推荐；
`targetResult` 只在 OptimizationResult 提供真实目标和状态字段时生成。
