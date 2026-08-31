import { escapeHtml, scientificText, deploymentText } from './formatters.js?v=visual1';

let charts = [];

function destroyCharts() {
  charts.forEach((chart) => chart.destroy?.());
  charts = [];
}

function metricSeries(metrics, metric, prefix) {
  return ['overall', 'low_load', 'ramp_down'].map((regime) => {
    const value = Number(metrics[`${regime}_${prefix}_${metric}`]);
    return Number.isFinite(value) ? value : null;
  });
}

function hasValues(values) {
  return values.some((value) => value !== null);
}

export async function renderResearchVisualization(container, history, reportLoader) {
  destroyCharts();
  const items = Array.isArray(history) ? history : [];
  const completed = items.filter((item) => item.status === 'completed');
  let report = null;
  let reportRun = null;
  for (const item of completed.slice(0, 5)) {
    try { report = await reportLoader(item.runId); reportRun = item; break; } catch { /* 尝试下一个具备报告的真实任务 */ }
  }

  const metrics = report?.verification_result?.metrics || {};
  const modelKey = metrics.selected_model_key || 'torch_dlinear_30';
  const maeModel = metricSeries(metrics, 'mae_m3_s', modelKey);
  const maeBase = metricSeries(metrics, 'mae_m3_s', 'persistence');
  const rmseModel = metricSeries(metrics, 'rmse_m3_s', modelKey);
  const rmseBase = metricSeries(metrics, 'rmse_m3_s', 'persistence');
  const hasMetrics = hasValues([...maeModel, ...maeBase, ...rmseModel, ...rmseBase]);
  const summary = report?.summary || {};

  container.innerHTML = `
    <section class="visual-summary">
      <div><span>历史任务</span><strong>${items.length}</strong></div>
      <div><span>已完成</span><strong>${completed.length}</strong></div>
      <div><span>最新科学结论</span><strong>${escapeHtml(scientificText(summary.scientific_status))}</strong></div>
      <div><span>工程判断</span><strong>${escapeHtml(deploymentText(summary.deployment_status))}</strong></div>
    </section>
    <section class="visual-grid">
      ${hasMetrics ? `<article class="visual-panel visual-panel-wide"><header><h3>最新实验模型对照</h3><p>${escapeHtml(reportRun?.question || report?.run_id || '')}</p></header><div class="chart-frame"><canvas id="latestMetricChart"></canvas></div></article>` : `<article class="visual-panel visual-panel-wide"><div class="feature-empty"><h3>最新报告没有可绘制指标</h3><p>保留空状态，不生成模拟预测曲线。</p></div></article>`}
    </section>`;

  if (!window.Chart) return;
  const common = { responsive: true, maintainAspectRatio: false, animation: { duration: 800 }, plugins: { legend: { labels: { color: '#b9b9b9', usePointStyle: true } } }, scales: { x: { ticks: { color: '#909090' }, grid: { color: '#333333' } }, y: { beginAtZero: true, ticks: { color: '#909090' }, grid: { color: '#333333' } } } };
  if (hasMetrics) charts.push(new Chart(document.querySelector('#latestMetricChart'), { type: 'bar', data: { labels: ['整体 MAE', '低负荷 MAE', '降负荷 MAE', '整体 RMSE', '低负荷 RMSE', '降负荷 RMSE'], datasets: [{ label: metrics.selected_model || '候选模型', data: [...maeModel, ...rmseModel], backgroundColor: '#8ea6b5' }, { label: 'Persistence', data: [...maeBase, ...rmseBase], backgroundColor: '#606060' }] }, options: common }));
}
