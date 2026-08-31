const stageOrder = ['problem', 'evidence', 'plan', 'execution', 'evaluation', 'report'];

function stageIndex(raw) {
  const stage = raw.stage || 'queued';
  if (stage === 'queued') return 0;
  if (['rag_completed', 'hypotheses_completed', 'elo_running', 'elo_completed'].includes(stage)) return 1;
  if (['research_plan_completed', 'needs_human_review'].includes(stage)) return 2;
  if (stage === 'completed' || stage === 'failed') return 5;
  if (raw.verification?.executed || raw.verification_result?.executed) return 4;
  return 3;
}

function statusFor(index, current, terminal) {
  if (terminal && index <= current) return 'completed';
  if (index < current) return 'completed';
  if (index === current) return 'active';
  return 'waiting';
}

function asArray(value) {
  return Array.isArray(value) ? value : [];
}

function firstValue(...values) {
  return values.find((value) => value !== undefined && value !== null && value !== '');
}

function solutionLabel(value) {
  if (!value) return '';
  if (typeof value === 'string') return value;
  return value.name || value.title || value.solution || value.solution_id || value.id || '';
}

function normalizeFrontend(raw, fallbackQuestion) {
  const run = raw.run || {};
  const execution = raw.execution || {};
  const scientific = execution.scientific_result || {};
  const audit = execution.audit || {};
  const hypotheses = asArray(raw.hypotheses).map((item) => ({
    ...item,
    id: item.hypothesis_id,
    hypothesis: item.statement,
    rationale: item.mechanism_chain,
    reasoning: item.expected_observation
  }));
  const selectedHypothesis = hypotheses.find((item) => item.rank === 1) || hypotheses[0] || {};
  const selectedPlan = selectedHypothesis.experiment_plan || {};
  const metricRows = asArray(execution.rows);
  const selectedRow = metricRows.find((item) => item.model === execution.protocol_selected_model) || {};
  const persistenceRow = metricRows.find((item) => String(item.model).toLowerCase() === 'persistence') || {};
  const metricAliases = {
    MAE: ['MAE', 'mae_m3_s', 'mae_t_h', 'mae'],
    RMSE: ['RMSE', 'rmse_m3_s', 'rmse_t_h', 'rmse'],
    R2: ['R2', 'r2_m3_s', 'r2_t_h', 'r2'],
    MBE: ['MBE', 'mbe_m3_s', 'mbe_t_h', 'mbe']
  };
  const primaryKey = String(execution.primary_metric || 'MAE').toUpperCase();
  const wantedKeys = metricAliases[primaryKey] || [primaryKey];
  const primaryValue = (row, scope) => {
    const values = row[scope] || {};
    for (const key of wantedKeys) {
      if (values[key] !== undefined && values[key] !== null && Number.isFinite(Number(values[key]))) return Number(values[key]);
    }
    return null;
  };
  const selectedModelPrimary = primaryValue(selectedRow, 'locked_test') ?? primaryValue(selectedRow, 'validation');
  const persistencePrimary = primaryValue(persistenceRow, 'locked_test') ?? primaryValue(persistenceRow, 'validation');
  const environment = execution.environment || {};
  const dependencyVersions = environment.dependency_versions || {};
  const metrics = {
    selected_model: execution.protocol_selected_model,
    locked_test_best_model: execution.locked_test_best_model,
    primary_metric: execution.primary_metric,
    metric_unit: execution.metric_unit,
    validation_metrics: selectedRow.validation || {},
    locked_test_metrics: selectedRow.locked_test || {},
    model_rows: metricRows,
    sample_count: selectedRow.sample_counts?.train ?? selectedPlan.sample_counts?.train,
    random_seed: selectedPlan.random_seed ?? selectedRow.random_seed,
    experiment_adapter: environment.adapter || selectedPlan.execution_backend || 'real_sklearn',
    python_version: environment.python_version,
    numpy_version: dependencyVersions.numpy,
    pandas_version: dependencyVersions.pandas,
    sklearn_version: dependencyVersions['scikit-learn'],
    torch_version: dependencyVersions.torch,
    selected_model_primary: selectedModelPrimary,
    persistence_primary: persistencePrimary
  };
  const status = run.status || 'queued';
  const stageIndex = Math.max(0, stageOrder.indexOf(run.current_stage));
  return {
    raw,
    runId: run.run_id,
    sessionId: null,
    question: run.question || fallbackQuestion,
    rawStage: status,
    progressPercent: Number(run.progress_percent) || 0,
    processRunning: ['queued', 'running'].includes(status),
    message: asArray(raw.errors).join('；'),
    errors: asArray(raw.errors),
    revision: run.revision,
    artifacts: asArray(raw.artifacts),
    currentStage: stageIndex,
    stages: asArray(raw.stages).map((stage) => ({
      id: stage.stage_id,
      title: stage.name,
      summary: stage.summary || '',
      status: stage.status === 'running' ? 'active' : stage.status
    })),
    problem: raw.problem || {},
    planBook: {},
    researchPlan: { ...selectedPlan, selection_reason: selectedHypothesis.selection_reason || '' },
    competitionPlan: { rankings: hypotheses },
    hypothesisCandidates: hypotheses,
    eloResult: { rankings: hypotheses },
    evidence: asArray(raw.evidence_summary?.items),
    evidenceNote: raw.evidence_summary?.degraded_note || '',
    evidenceLocalStats: raw.evidence_summary?.local_stats || null,
    evidenceDegradedCandidates: asArray(raw.evidence_summary?.degraded_candidates),
    selectedHypothesis: selectedHypothesis.statement || '等待生成',
    selectedSolution: selectedPlan,
    selectedSolutionLabel: execution.protocol_selected_model || '等待选择',
    baselines: selectedPlan.reference_models || ['persistence'],
    modelFamily: execution.protocol_selected_model || '等待选择',
    guard: audit,
    leakageStatus: audit.leakage_check_passed === true ? '已通过泄漏检查' : audit.leakage_check_passed === false ? '泄漏检查未通过' : '等待后端检查',
    verification: {
      ...scientific,
      ...audit,
      metrics,
      executed: Boolean(execution.experiment_id),
      executed_step_ids: asArray(execution.executed_step_ids)
    },
    control: raw.control || null,
    metrics,
    modelRows: metricRows,
    selectionRationale: execution.selection_rationale || null,
    selectionDetail: execution.selection_detail || null,
    rankingTrace: raw.ranking || null,
    datasetIsReal: Boolean(selectedPlan.dataset_id || selectedPlan.dataset_path),
    scientificStatus: scientific.verdict,
    deployment: {},
    deploymentStatus: audit.execution_valid === true ? 'scientifically_valid' : null,
    elapsedSeconds: null,
    needsHumanReview: status === 'needs_human_review',
    failed: status === 'failed',
    completed: ['completed', 'completed_with_warning'].includes(status),
    experimentRows: metricRows.map((row) => ({
      id: row.model,
      title: row.model,
      detail: row.fit_success === false ? row.failure_reason || '执行失败' : `Validation 与 Locked test 指标已记录`,
      status: row.fit_success === false ? 'failed' : 'complete'
    }))
  };
}

export function normalizeResearch(raw, fallbackQuestion = '') {
  if (raw?.schema_version === 'boilermind.frontend.research_run.v1') {
    return normalizeFrontend(raw, fallbackQuestion);
  }
  const current = stageIndex(raw);
  const terminal = ['completed', 'failed'].includes(raw.stage);
  const finalReport = raw.final_report || {};
  const verification = raw.verification || raw.verification_result || finalReport.verification_result || {};
  const metrics = verification.metrics || {};
  const planBook = raw.plan_book || {};
  const researchPlan = raw.research_plan || finalReport.research_plan || {};
  const problem = raw.problem_decomposition || finalReport.problem_decomposition || {};
  const competitionPlan = raw.experiment_plan || {};
  const selectedSolution = firstValue(
    researchPlan.selected_solution,
    planBook.technical_details?.selected_solution,
    competitionPlan.selected_solution,
    raw.selected_solution,
    {}
  );
  const hypothesisCandidates = asArray(raw.hypothesis_candidates).length ? raw.hypothesis_candidates : asArray(researchPlan.hypothesis_candidates);
  const eloResult = raw.elo_result || {};
  const evidence = asArray(planBook.references).length ? planBook.references : asArray(raw.rag_result?.references || raw.rag_result?.evidence);
  const artifacts = asArray(raw.artifacts).length ? raw.artifacts : asArray(verification.artifacts);
  const scientificStatus = verification.scientific_status || raw.hypothesis_evaluation?.scientific_status || finalReport.hypothesis_evaluation?.scientific_status || finalReport.summary?.scientific_status;
  const deployment = raw.deployment_gate_report || finalReport.deployment_gate_report || {};
  const deploymentStatus = deployment.deployment_status || finalReport.summary?.deployment_status;
  const selectedHypothesis = firstValue(researchPlan.selected_hypothesis, planBook.scientific_hypothesis, eloResult.rankings?.[0]?.hypothesis, hypothesisCandidates[0]?.hypothesis, '等待生成');
  const baselines = asArray(researchPlan.experiment_contract?.baselines || planBook.experiments?.baselines || researchPlan.control_groups);
  const modelFamily = firstValue(researchPlan.model_selection?.selected_model, planBook.technical_details?.model_family, metrics.selected_model, solutionLabel(selectedSolution), '等待选择');
  const guard = raw.research_plan_guard || finalReport.research_plan_guard || researchPlan.experiment_contract?.guard || {};
  const elapsedSeconds = Number(metrics.total_elapsed_seconds);
  const datasetIsReal = metrics.dataset_kind === 'real_boiler_historical_data' || String(metrics.real_experiment_executed) === 'true' || verification.execution_mode === 'real';
  const leakageStatus = firstValue(
    guard.approved === true || String(metrics.guard_approved) === 'true' ? '已通过研究门禁' : '',
    guard.approved === false ? '门禁未通过' : '',
    '等待后端检查'
  );
  const stageDefinitions = [
    ['problem', '问题拆解', '研究目标、变量、工况与时间范围'],
    ['evidence', '证据与假设', '文献证据、候选假设与 Elo 比较'],
    ['plan', '实验方案', '选定方案、基线、模型与执行门禁'],
    ['execution', '实验执行', '当前任务、真实数据、耗时与产物'],
    ['evaluation', '科学评价', '假设结论与关键对比指标'],
    ['report', '科研报告', '报告生成状态与阅读入口']
  ];
  const question = raw.research_question || raw.question || problem.research_question || fallbackQuestion;
  return {
    raw,
    runId: raw.run_id || raw.runId,
    sessionId: raw.session_id || raw.sessionId,
    question,
    rawStage: raw.stage || 'queued',
    processRunning: Boolean(raw.process_running),
    message: raw.message || '',
    artifacts,
    currentStage: current,
    stages: stageDefinitions.map(([id, title, summary], index) => ({ id, title, summary, status: statusFor(index, current, terminal) })),
    problem,
    planBook,
    researchPlan,
    competitionPlan,
    hypothesisCandidates,
    eloResult,
    evidence,
    selectedHypothesis,
    selectedSolution,
    selectedSolutionLabel: solutionLabel(selectedSolution) || competitionPlan.selected_solution_id || '等待选择',
    baselines,
    modelFamily,
    guard,
    leakageStatus,
    verification,
    metrics,
    datasetIsReal,
    scientificStatus,
    deployment,
    deploymentStatus,
    elapsedSeconds: Number.isFinite(elapsedSeconds) ? elapsedSeconds : null,
    needsHumanReview: raw.stage === 'needs_human_review',
    failed: raw.stage === 'failed',
    completed: raw.stage === 'completed',
    experimentRows: [
      { id: 'data', title: '数据准备', detail: metrics.sample_count ? `真实锅炉历史数据 · ${Number(metrics.sample_count).toLocaleString('zh-CN')}条样本` : '等待数据准备产物', status: metrics.sample_count || current > 3 || verification.executed ? 'complete' : current === 3 ? 'running' : 'waiting' },
      { id: 'model', title: '模型实验', detail: metrics.selected_model ? `${metrics.selected_model} 与 Persistence 对照` : modelFamily, status: verification.executed ? 'complete' : current === 3 ? 'running' : 'waiting' },
      { id: 'test', title: '锁定测试', detail: metrics.overall_sample_count ? `锁定测试集 · ${Number(metrics.overall_sample_count).toLocaleString('zh-CN')}条样本` : '等待模型完成', status: verification.executed ? 'complete' : 'waiting' },
      { id: 'summary', title: '结果汇总', detail: scientificStatus ? `科学状态：${scientificStatus}` : '等待测试结果', status: scientificStatus ? 'complete' : 'waiting' }
    ]
  };
}

export function humanStage(stage) {
  const map = { queued: '等待开始', running: '运行中', rag_completed: '证据检索完成', hypotheses_completed: '假设已生成', elo_running: '正在比较假设', elo_completed: '假设排序完成', research_plan_completed: '实验方案完成', experiment_running: '运行中', verification_running: '锁定测试中', reporting: '正在生成报告', needs_human_review: '等待人工复核', completed: '已完成', failed: '执行失败' };
  return map[stage] || stage;
}

export { stageOrder };
