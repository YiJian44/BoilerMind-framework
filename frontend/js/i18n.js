// frontend/js/i18n.js
//
// Display-layer translations for backend enum values. The strings returned by
// the FastAPI research API stay in English (Python StrEnum values), so any
// tweak here never breaks the JSON contract. These helpers are used in views
// / app / formatters to render human-readable Chinese labels.

const RESEARCH_RUN_STATUS = {
  created: '已创建',
  problem_parsed: '问题已解析',
  evidence_retrieved: '证据已检索',
  evidence_verified: '证据已验证',
  evidence_frozen: '证据已冻结',
  hypotheses_generated: '假设已生成',
  hypotheses_qualified: '假设已筛选',
  prior_ranked: '已先验排序',
  top3_planned: 'Top-3 已规划',
  champion_testing: '正测试候选',
  result_audited: '结果已审计',
  dynamic_reranked: '已动态重排',
  extended_validation: '扩展验证中',
  resolved: '已解决',
  unresolved_best_effort: '尽力未解',
  knowledge_updated: '知识已更新',
  report_ready: '报告已生成',
  failed: '执行失败',
  queued: '等待开始',
  running: '运行中',
  completed: '已完成',
  completed_with_warning: '已完成(报告告警)',
  needs_human_review: '等待人工复核',
};

const HYPOTHESIS_STATUS = {
  generated: '已生成',
  qualified: '已纳入候选',
  rejected: '已剔除',
  planned: '已规划',
  testing: '测试中',
  supported: '得到支持',
  partially_supported: '部分支持',
  falsified: '被证伪',
  insufficient_evidence: '证据不足',
};

const SCIENTIFIC_VERDICT = {
  supported: '得到支持',
  partially_supported: '部分支持',
  falsified: '被证伪',
  insufficient_evidence: '证据不足',
};

const EVIDENCE_STAGE = {
  retrieved: '已检索',
  candidate: '候选',
  verified: '已验证',
  rejected: '已拒绝',
};

const CLAIM_SUPPORT = {
  direct: '直接支持',
  partial: '部分支持',
  contradicting: '反向证据',
  irrelevant: '不相关',
  unknown: '待定',
};

const APPLICABILITY_LEVEL = {
  high: '高',
  medium: '中',
  low: '低',
  unknown: '未知',
};

const EXPERIMENT_STATUS = {
  planned: '已规划',
  running: '运行中',
  completed: '已完成',
  invalid: '无效',
  failed: '执行失败',
};

const RESEARCH_STOP_REASON = {
  target_metric_reached: '达到目标指标',
  hypothesis_resolved: '假设已解决',
  all_candidates_exhausted: '候选已穷尽',
  max_rounds_reached: '达到最大轮次',
  time_budget_exhausted: '时间预算耗尽',
  no_meaningful_improvement: '无显著提升',
  capability_blocked: '能力受限',
  execution_failed: '执行失败',
};

const PROBLEM_RESOLUTION_STATUS = {
  solved: '已解决',
  partially_solved: '部分解决',
  not_solved: '未解决',
  insufficient_evidence: '证据不足',
  execution_blocked: '执行被阻断',
};

const MECHANISM_SUPPORT_TYPE = {
  verified_evidence: '已验证证据',
  data_observation: '数据观察',
  domain_prior: '领域先验',
  hypothesis_inference: '假设推理',
};

const UNITY_PUSH_STATUS = {
  none: '尚未推送',
  payload_generated: '仅生成指令',
  sent: '已发送',
  received: '已接收',
  executed: '已执行',
  returned: '已回传',
};

const UNITY_SECOND_VERDICT = {
  supported: '支持',
  partially_supported: '部分支持',
  falsified: '证伪',
};

const CONTROL_CONCLUSION_SCOPE = {
  small_model_control_validation: '小模型控制优化验证',
};

// scikit-learn / experimental model aliases. Lowercase keys, original cases
// fall through to the user-supplied string so newly-added model names still
// render.
const MODEL_NAME = {
  persistence: '持久性基线模型',
  ridge: 'Ridge',
  bayesianridge: 'BayesianRidge',
  rf: 'RandomForest',
  randomforest: 'RandomForest',
  hgb: 'HistGradientBoosting',
  mlp: 'MLP',
  svr: 'SVR',
  elasticnet: 'ElasticNet',
  pls: 'PLS',
  knn: 'KNN',
  bayesian_ridge: 'BayesianRidge',
  gpr: 'GPR',
  lstm: 'LSTM',
  gru: 'GRU',
  transformer: 'Transformer',
  dlinear: 'DLinear',
  // control-optimization runner alias from the bundled experiment backend
  hgb_control_optimizer: 'HGB 控制优化器',
  current_operating_point: '当前工况点',
  constrained_control_optimization: '受约束控制优化',
};

export function t(value, table) {
  if (value === null || value === undefined) return '';
  const key = String(value);
  const map = table || {};
  return Object.prototype.hasOwnProperty.call(map, key) ? map[key] : value;
}

export const researchRunStatus = (value) => t(value, RESEARCH_RUN_STATUS);
export const hypothesisStatus = (value) => t(value, HYPOTHESIS_STATUS);
export const scientificVerdict = (value) => t(value, SCIENTIFIC_VERDICT);
// Backwards compatible alias used by views.js verdictLabel().
export const verdictLabel = scientificVerdict;
export const evidenceStage = (value) => t(value, EVIDENCE_STAGE);
export const claimSupport = (value) => t(value, CLAIM_SUPPORT);
export const applicabilityLevel = (value) => t(value, APPLICABILITY_LEVEL);
export const experimentStatus = (value) => t(value, EXPERIMENT_STATUS);
export const researchStopReason = (value) => t(value, RESEARCH_STOP_REASON);
export const problemResolutionStatus = (value) => t(value, PROBLEM_RESOLUTION_STATUS);
export const mechanismSupportType = (value) => t(value, MECHANISM_SUPPORT_TYPE);
export const unityPushStatus = (value) => t(value, UNITY_PUSH_STATUS);
export const unityStatusLabels = UNITY_PUSH_STATUS;
export const unitySecondVerdict = (value) => t(value, UNITY_SECOND_VERDICT);
export const controlConclusionScope = (value) => t(value, CONTROL_CONCLUSION_SCOPE);
export const modelName = (value) => {
  if (value === null || value === undefined) return '';
  return t(String(value).toLowerCase(), MODEL_NAME);
};

// Some backend values arrive as uppercase legacy enum names (e.g. "SUPPORTED",
// "FALSIFIED"). Try the lower-case key, then the raw key, then fall back.
export function verdictLabelAny(value) {
  if (value === null || value === undefined) return '';
  const upper = String(value).toUpperCase();
  if (Object.prototype.hasOwnProperty.call(SCIENTIFIC_VERDICT, upper.toLowerCase())) {
    return SCIENTIFIC_VERDICT[upper.toLowerCase()];
  }
  if (Object.prototype.hasOwnProperty.call(SCIENTIFIC_VERDICT, upper)) {
    return SCIENTIFIC_VERDICT[upper];
  }
  return value;
}

export const I18N_TABLES = {
  RESEARCH_RUN_STATUS,
  HYPOTHESIS_STATUS,
  SCIENTIFIC_VERDICT,
  EVIDENCE_STAGE,
  CLAIM_SUPPORT,
  APPLICABILITY_LEVEL,
  EXPERIMENT_STATUS,
  RESEARCH_STOP_REASON,
  PROBLEM_RESOLUTION_STATUS,
  MECHANISM_SUPPORT_TYPE,
  UNITY_PUSH_STATUS,
  UNITY_SECOND_VERDICT,
  MODEL_NAME,
};
