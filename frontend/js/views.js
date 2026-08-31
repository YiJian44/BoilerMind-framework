import { escapeHtml, safeRichText, formatTime, shortTitle, statusText, scientificText, deploymentText, semanticMetricEntries, formatMetricDisplay, rawMetricsJson } from './formatters.js?v=20260826-kg9';
import { humanStage } from './research-adapter.js';
import {
  verdictLabel, scientificVerdict, hypothesisStatus, evidenceStage, claimSupport,
  applicabilityLevel, experimentStatus, researchStopReason, problemResolutionStatus,
  mechanismSupportType, unityPushStatus, unitySecondVerdict, modelName as i18nModelName,
  researchRunStatus, unityStatusLabels, controlConclusionScope,
} from './i18n.js';

const icon = (name, alt = '') => `<img src="assets/icons/${name}.svg" alt="${escapeHtml(alt)}">`;

export const executionMetricsStore = new Map();
const iterationLiveState = new Map();

export function renderHistoryList(container, items) {
  container.innerHTML = items.length ? items.map((item) => `
    <button class="conversation-item" type="button" data-run-id="${escapeHtml(item.runId)}">
      <strong>${escapeHtml(shortTitle(item.question, item.title))}</strong>
      <span>${escapeHtml(statusText(item.status))} · ${escapeHtml(formatTime(item.updatedAt))}</span>
    </button>`).join('') : '<div class="empty-state">还没有研究记录</div>';
}

export function renderHistoryTable(container, items) {
  container.innerHTML = items.length ? items.map((item) => `
    <div class="history-row">
      <button type="button" data-run-id="${escapeHtml(item.runId)}"><strong>${escapeHtml(shortTitle(item.question, item.title))}</strong><small>${escapeHtml(item.question || item.runId)}</small></button>
      <span class="status-label ${escapeHtml(item.status)}">${escapeHtml(statusText(item.status))}</span>
      <span>${escapeHtml(formatTime(item.updatedAt))}</span>
    </div>`).join('') : '<div class="empty-state">没有符合条件的历史研究</div>';
}

export function renderRunning(container, activeRuns, pendingRequests = new Map()) {
  const pending = [...pendingRequests.values()];
  const runs = [...activeRuns.values()].filter((run) => !run.completed && !run.failed && (run.processRunning || run.rawStage === 'queued'));
  const cards = [
    ...pending.map((item) => `<article class="running-card"><div><span class="live-label">问答生成中</span><strong>${escapeHtml(shortTitle(item.question))}</strong><p>${escapeHtml(item.stage || '正在连接证据与研究上下文')}</p></div><button class="secondary-button" type="button" data-action="return-pending" data-request-id="${escapeHtml(item.id)}">返回对话</button></article>`),
    ...runs.map((run) => `<article class="running-card"><div><span class="live-label">真实研究</span><strong>${escapeHtml(shortTitle(run.question))}</strong><p>${escapeHtml(humanStage(run.rawStage))} · 六阶段中的第 ${run.currentStage + 1} 阶段</p></div><button class="secondary-button" type="button" data-action="return-active-run" data-run-id="${escapeHtml(run.runId)}">返回研究</button></article>`)
  ];
  container.innerHTML = cards.length ? cards.join('') : '<div class="empty-state">当前没有进行中的问题或实验。</div>';
}

export function appendUserMessage(container, question, mode) {
  const element = document.createElement('article');
  element.className = 'message user-message';
  element.innerHTML = `<div class="message-meta">你 · ${mode === 'research' ? '直接研究' : '对话'}</div><div>${escapeHtml(question)}</div>`;
  container.append(element);
  return element;
}

export function appendLoadingMessage(container, label = '正在形成工程回答') {
  const element = document.createElement('article');
  element.className = 'message assistant-message';
  element.innerHTML = `<div class="message-meta">BoilerMind</div><h3>${escapeHtml(label)}</h3><p class="assistant-copy">正在连接证据与研究上下文，请稍候……</p>`;
  container.append(element);
  return element;
}

export function appendAssistantProgress(container, question) {
  const element = document.createElement('article');
  element.className = 'message assistant-progress';
  element.innerHTML = `<div class="message-meta">BoilerMind · 后端研究链</div><div class="assistant-progress-head"><div><h3>正在检索证据并形成研究问题</h3><p>${escapeHtml(question)}</p></div><span class="live-label"><span class="live-dot"></span><span data-progress-elapsed>0 秒</span></span></div><div class="assistant-progress-steps"><div class="completed"><span>1</span><div><strong>请求已发送</strong><small>研究问题已提交至后端</small></div></div><div class="active"><span>2</span><div><strong>证据检索与假设生成</strong><small>后端正在运行，接口为非流式响应</small></div></div><div><span>3</span><div><strong>结构化研究问题</strong><small>返回后自动进入六阶段完整实验</small></div></div></div>`;
  container.append(element);
  return element;
}

export function renderAssistantMessage(element, response) {
  const dataNeeds = response.data_needs || [];
  const ready = Boolean(response.hypothesis_ready);
  const canResearch = Boolean(response.research_question_summary?.trim());
  element.innerHTML = `
    <div class="message-meta">BoilerMind · ${escapeHtml(response.provider || 'engineering')}</div>
    <div class="assistant-copy">${safeRichText(response.answer || '后端没有返回可用回答')}</div>
    ${dataNeeds.length ? `<div class="detail-block"><h3>建议补充的数据</h3><p>${dataNeeds.map(escapeHtml).join('；')}</p></div>` : ''}
    ${canResearch ? `<div class="message-actions">
      ${canResearch ? `<span class="research-handoff"><span class="live-dot"></span>${ready ? '证据与假设已就绪' : '已形成可执行研究问题'}，正在自动进入完整实验</span>` : ''}
    </div>` : ''}`;
}

export function appendResearchLaunch(container, question) {
  const element = document.createElement('article');
  element.className = 'message research-launch';
  element.innerHTML = `<div class="message-meta">BoilerMind · 完整研究</div><div class="research-launch-head"><div><h3>正在建立六阶段实验流程</h3><p>${escapeHtml(question)}</p></div><span class="live-label">准备中</span></div><div class="launch-stages">${['问题拆解', '证据与假设', '实验方案', '实验执行', '科学评价', '科研报告'].map((title, index) => `<div class="${index === 0 ? 'active' : ''}"><span>${index + 1}</span><strong>${title}</strong><small>${index === 0 ? '正在创建研究任务' : '等待前序阶段'}</small></div>`).join('')}</div>`;
  container.append(element);
  return element;
}

export function renderAssistantError(element, error) {
  element.innerHTML = `<div class="message-meta">BoilerMind · 连接异常</div><h3>本次回答未完成</h3><p class="assistant-copy">${escapeHtml(error.message)}</p><div class="message-actions"><button class="secondary-button" type="button" data-action="return-current-conversation">返回对话</button>${element.dataset.question ? `<button class="research-button" type="button" data-action="retry-assistant" data-question="${escapeHtml(element.dataset.question)}">重新请求</button>` : ''}</div>`;
}

function executionIcon(id) {
  return ({ data: 'database', model: 'beaker', test: 'shield-check', summary: 'file-earmark-text' })[id] || 'circle';
}

function textValue(value, fallback = '后端尚未提供') {
  if (value === undefined || value === null || value === '') return fallback;
  if (typeof value === 'object') return value.name || value.title || value.id || JSON.stringify(value);
  return localizeUiText(i18nModelName(String(value)));
}

function localizeUiText(value) {
  return String(value || '')
    .replace(/chronological_train_validation_locked_test/gi, '按时间顺序划分训练集、验证集与锁定测试集')
    .replace(/regime_stratified_evaluation/gi, '工况分层评估')
    .replace(/reference_model_comparison/gi, '参考基线对比')
    .replace(/model_comparison/gi, '模型对比')
    .replace(/chronological_validation/gi, '时序验证')
    .replace(/locked_test_evaluation/gi, '锁定测试集评估')
    .replace(/projection_detect/gi, '投影检测')
    .replace(/dynamic_score/gi, '动态评分')
    .replace(/Locked-test/gi, '锁定测试集')
    .replace(/Locked test/gi, '锁定测试集')
    .replace(/Validation/gi, '验证集')
    .replace(/FALSIFIED/gi, '已证伪')
    .replace(/SUPPORTED/gi, '已支持')
    .replace(/Persistence/gi, '持久性基线模型');
}

function numberValue(value, digits = 3) {
  const number = Number(value);
  return Number.isFinite(number) ? number.toLocaleString('zh-CN', { maximumFractionDigits: digits }) : '—';
}

function constraintText(value) {
  const text = textValue(value);
  const pressure = text.match(/^drum_pressure\s*<=\s*([\d.]+)\s*MPa$/i);
  return pressure ? `汽包压力 ≤ ${pressure[1]} MPa` : text;
}

function infoRow(label, value, className = '') {
  return `<div class="research-info-row ${className}"><span>${escapeHtml(label)}</span><strong>${escapeHtml(textValue(value))}</strong></div>`;
}

function corpusLevelLabel(level) {
  return {
    core: '核心文献',
    domain_support: '领域支撑文献',
    method_support: '方法支撑文献',
  }[level] || level;
}

function renderHypothesisFocus(hypothesis) {
  return `<section class="hypothesis-focus-card">
    <div class="hypothesis-focus-head"><span>本轮核心假设</span><i>证据与假设</i></div>
    <p>${escapeHtml(hypothesis || '等待生成')}</p>
  </section>`;
}

function renderEvidenceDetails(research) {
  const references = research.evidence.slice(0, 12);
  const hypotheses = research.hypothesisCandidates.slice(0, 8);
  const localStats = research.evidenceLocalStats;
  const localStatsBlock = localStats ? `<div class="detail-block"><h3>本地文献库可用（未执行语义核验）</h3><p>本地库共 ${escapeHtml(localStats.paper_count)} 篇 / ${escapeHtml(localStats.chunk_count)} 个文本片段，分层：${Object.entries(localStats.by_corpus_level || {}).map(([level, count]) => `${escapeHtml(corpusLevelLabel(level))} ${escapeHtml(count)} 篇`).join('、')}。以下候选摘要由本地检索得到，仅供查看，未经过语义核验。</p>${research.evidenceDegradedCandidates.length ? research.evidenceDegradedCandidates.slice(0, 5).map((item) => `<article class="evidence-item"><div><strong>${escapeHtml(item.title || item.citation || '候选')}</strong><p>${escapeHtml(item.snippet || '')}</p></div></article>`).join('') : '<p class="muted-copy">暂无候选摘要。</p>'}</div>` : '';
  return `<div class="stage-glance">
      ${infoRow('文献证据', `${research.evidence.length} 条`)}
      ${infoRow('科研假设', `${hypotheses.length} 条`)}
    </div>
    ${renderHypothesisFocus(research.selectedHypothesis)}
    ${research.evidenceNote ? `<div class="detail-block"><h3>文献检索降级说明</h3><p>${escapeHtml(research.evidenceNote)}</p></div>` : ''}
    ${localStatsBlock}
    <details class="research-detail"><summary>文献证据详情 <span>${research.evidence.length}</span></summary><div class="research-detail-body">
      ${references.length ? references.map((item, index) => `<article class="evidence-item"><div><strong>${escapeHtml(item.title || `证据 ${index + 1}`)}</strong><p>${escapeHtml(item.formatted_citation || item.doi_or_url || item.citation || item.evidence_source || item.evidence_id || '')}</p></div><span class="source-badge">${item.citation_verified && item.semantic_verified ? '已验证' : escapeHtml(item.verification_status || '来源已记录')}</span></article>`).join('') : '<p class="muted-copy">当前阶段尚未返回文献清单。</p>'}
    </div></details>
    <details class="research-detail"><summary>科研假设与实验依据 <span>${hypotheses.length}</span></summary><div class="research-detail-body hypothesis-list">
      ${hypotheses.length ? hypotheses.map((item, index) => `<article><span>${escapeHtml(item.hypothesis_id || `H${index + 1}`)}</span><div><strong>${escapeHtml(item.title || item.hypothesis || '候选假设')}</strong><p>${escapeHtml(item.hypothesis || '')}</p><small>${escapeHtml(item.rationale || item.reasoning || '')}</small>${item.experiment_plan ? `<details><summary>对应实验规划书</summary><div>${infoRow('实验类型', item.experiment_plan.experiment_type)}${infoRow('候选模型', (item.experiment_plan.candidate_models || []).join('、'))}${infoRow('参考模型', (item.experiment_plan.reference_models || []).join('、'))}${infoRow('主指标', item.experiment_plan.primary_metric)}${infoRow('预测时域', `${item.experiment_plan.prediction_horizon_steps ?? '—'} 步`)}</div></details>` : ''}</div></article>`).join('') : '<p class="muted-copy">等待候选假设产物。</p>'}
    </div></details>
    ${research.rankingTrace ? `${renderRankingTable(research)}${renderIterationReplaySlot(research)}` : ''}`;
}

function renderRankingTable(research) {
  const trace = research.rankingTrace;
  const rounds = trace?.rounds || [];
  const last = rounds[rounds.length - 1];
  if (!last) return '';
  const rows = last.entries.map((entry) => {
    const info = (research.hypothesisCandidates || []).find((item) => item.hypothesis_id === entry.hypothesis_id) || {};
    const status = localizeUiText(info.status ? verdictLabel(info.status) : (entry.eligible ? '候选' : '已淘汰'));
    const feedback = entry.cumulative_feedback;
    const feedbackText = feedback == null ? '—' : `${feedback > 0 ? '+' : ''}${numberValue(feedback, 2)}`;
    const droppedReason = localizeUiText((entry.dropped_reasons || []).join('、'));
    return `<article class="ranking-card ${entry.eligible ? 'eligible' : 'dropped'}">
      <header><div><span>候选假设</span><strong>${escapeHtml(entry.hypothesis_id)}</strong></div><b>${escapeHtml(status)}</b></header>
      <div class="ranking-score-grid">
        <div><span>历史支持</span><strong>${numberValue(entry.historical_support, 2)}</strong></div>
        <div><span>先验分</span><strong>${numberValue(entry.prior_score, 3)}</strong></div>
        <div><span>累计反馈</span><strong>${escapeHtml(feedbackText)}</strong></div>
        <div class="ranking-dynamic-score"><span>动态分</span><strong>${numberValue(entry.dynamic_score, 3)}</strong></div>
      </div>
      ${droppedReason ? `<p>淘汰原因：${escapeHtml(droppedReason)}</p>` : ''}
    </article>`;
  }).join('');
  return `<section class="ranking-table-section"><span class="overline">假设排序</span><h4>假设排序与历史反馈</h4><p class="ranking-intro">以下为本轮最终排序；动态分综合了先验支持与本轮实验反馈。</p><div class="ranking-card-list">${rows}</div></section>`;
}

function renderIterationReplaySlot(research) {
  const trace = research.rankingTrace;
  if (!trace?.rounds?.length) return '';
  return `<section class="iteration-replay" data-iteration-replay>
    <div class="iteration-replay-head">
      <div><span class="overline">排序回放</span><h4>假设重排回放</h4></div>
      <div class="iteration-controls">
        <button type="button" data-iter-action="restart" title="回到初始排序">↺</button>
        <button type="button" data-iter-action="prev" title="上一步">‹</button>
        <button type="button" data-iter-action="playpause" title="播放 / 暂停">▶</button>
        <button type="button" data-iter-action="next" title="下一步">›</button>
        <select data-iter-speed title="播放速度"><option value="1600">1×</option><option value="800" selected>2×</option><option value="350">4×</option></select>
      </div>
    </div>
    <div class="iteration-round-label" data-iter-round-label>第 1 / ${trace.rounds.length} 轮</div>
    <div class="iteration-list" data-iter-list></div>
    <div class="iteration-log" data-iter-log></div>
  </section>`;
}

function renderIterationFeedback(research) {
  const trace = research.rankingTrace;
  if (!trace?.rounds?.length) return '';
  const rounds = trace.rounds || [];
  const byRound = new Map();
  (trace.feedback || []).forEach((item) => {
    if (!byRound.has(item.round_index)) byRound.set(item.round_index, []);
    byRound.get(item.round_index).push(item);
  });
  const lines = [...byRound.entries()].map(([roundIndex, items]) => {
    const parts = items.map((item) => {
      let delta = '';
      const cur = rounds[roundIndex]?.entries?.find((e) => e.hypothesis_id === item.hypothesis_id);
      const pre = roundIndex > 0 ? rounds[roundIndex - 1]?.entries?.find((e) => e.hypothesis_id === item.hypothesis_id) : null;
      if (cur && pre && cur.cumulative_feedback != null && pre.cumulative_feedback != null) {
        const d = cur.cumulative_feedback - pre.cumulative_feedback;
        if (Math.abs(d) > 1e-9) delta = `（反馈 ${d > 0 ? '+' : ''}${numberValue(d, 2)}）`;
      }
      return `${item.hypothesis_id} ${verdictLabel(item.verdict)}${delta}`;
    });
    return `<li><strong>第 ${roundIndex} 批</strong>：${parts.join('、')}</li>`;
  });
  return `<section class="iteration-feedback"><span class="overline">FEEDBACK LOOP</span><h4>本轮迭代反馈</h4><ul class="iteration-feedback-list">${lines.join('')}</ul></section>`;
}

function animateNumber(el, from, to, duration) {
  if (!el) return;
  const start = performance.now();
  const step = (now) => {
    const t = Math.min(1, (now - start) / duration);
    const eased = 1 - Math.pow(1 - t, 3);
    el.textContent = numberValue(from + (to - from) * eased, 3);
    if (t < 1) requestAnimationFrame(step);
  };
  requestAnimationFrame(step);
}

export function captureIterationState(element, runId) {
  if (!element?.querySelectorAll || !runId) return;
  const cards = new Map();
  element.querySelectorAll('.iter-card').forEach((card) => {
    cards.set(card.dataset.hid, {
      rect: card.getBoundingClientRect(),
      score: Number(card.dataset.score)
    });
  });
  const panel = element.querySelector('[data-iteration-replay]');
  const roundIndex = Number(panel?.dataset.iterRound || 0);
  iterationLiveState.set(runId, { cards, roundIndex });
}

export function initIterationReplay(root, research) {
  if (!root?.querySelector) return;
  const panel = root.querySelector('[data-iteration-replay]');
  if (!panel || panel.dataset.iterReady === '1') return;
  const trace = research.rankingTrace;
  if (!trace?.rounds?.length) return;
  panel.dataset.iterReady = '1';
  const list = panel.querySelector('[data-iter-list]');
  const label = panel.querySelector('[data-iter-round-label]');
  const logEl = panel.querySelector('[data-iter-log]');
  const playButton = panel.querySelector('[data-iter-action="playpause"]');
  const speedSelect = panel.querySelector('[data-iter-speed]');
  const rounds = trace.rounds;
  const hypInfo = new Map((research.hypothesisCandidates || []).map((item) => [item.hypothesis_id, item]));
  let roundIndex = 0;
  let playing = false;
  let timer = null;
  let speed = Number(speedSelect?.value || 800);
  const runId = research.runId;
  const liveMode = Boolean(research.processRunning);
  const externalState = liveMode ? (iterationLiveState.get(runId) || null) : null;
  const externalPrev = externalState?.cards || null;
  if (!liveMode) iterationLiveState.delete(runId);

  const clearTimer = () => { if (timer) { clearInterval(timer); timer = null; } };
  const stopPlay = () => { playing = false; if (playButton) playButton.textContent = '▶'; clearTimer(); };
  const deltaFor = (index, hid) => {
    if (index <= 0) return null;
    const cur = rounds[index]?.entries?.find((e) => e.hypothesis_id === hid);
    const pre = rounds[index - 1]?.entries?.find((e) => e.hypothesis_id === hid);
    if (cur && pre && cur.cumulative_feedback != null && pre.cumulative_feedback != null) {
      const d = cur.cumulative_feedback - pre.cumulative_feedback;
      if (Math.abs(d) > 1e-9) return d;
    }
    return null;
  };
  const verdictFor = (index, hid) => {
    const item = (trace.feedback || []).find((f) => f.round_index === index && f.hypothesis_id === hid);
    return item ? item.verdict : null;
  };

  const renderRound = (index, animate = true, externalPrevCards = null) => {
    const snapshot = rounds[index];
    const entries = snapshot.entries || [];
    const isNewRound = externalState ? externalState.roundIndex !== index : true;
    const prevById = new Map();
    if (externalPrevCards) {
      externalPrevCards.forEach((value, hid) => prevById.set(hid, value));
    } else {
      [...list.querySelectorAll('.iter-card')].forEach((card) => {
        prevById.set(card.dataset.hid, {
          rect: card.getBoundingClientRect(),
          score: Number(card.dataset.score)
        });
      });
    }
    const sorted = [...entries].sort((a, b) => (Number(b.eligible) - Number(a.eligible)) || (b.dynamic_score - a.dynamic_score) || String(a.hypothesis_id).localeCompare(String(b.hypothesis_id)));
    const frag = document.createDocumentFragment();
    sorted.forEach((entry, position) => {
      const hid = entry.hypothesis_id;
      const info = hypInfo.get(hid) || {};
      const delta = deltaFor(index, hid);
      const verdict = verdictFor(index, hid);
      const status = localizeUiText(info.status ? verdictLabel(info.status) : (entry.eligible ? '候选' : '已淘汰'));
      const card = document.createElement('div');
      const flash = isNewRound ? (verdict === 'FALSIFIED' ? ' falsified-flash' : verdict === 'SUPPORTED' ? ' supported-flash' : '') : '';
      card.className = `iter-card${entry.eligible ? '' : ' dropped'}${flash}`;
      card.dataset.hid = hid;
      card.dataset.score = String(entry.dynamic_score == null ? 0 : entry.dynamic_score);
      card.innerHTML = `
        <span class="iter-rank">${position + 1}</span>
        <div class="iter-card-main"><strong>${escapeHtml(hid)}</strong><small>${escapeHtml(info.statement || info.title || '')}</small></div>
        <div class="iter-card-scores"><span class="iter-score" data-score>${numberValue(entry.dynamic_score, 3)}</span>${delta != null ? `<span class="iter-feedback ${delta > 0 ? 'pos' : 'neg'}">${delta > 0 ? '+' : ''}${numberValue(delta, 2)}</span>` : ''}<span class="iter-prior">先验 ${numberValue(entry.prior_score, 3)}</span></div>
        <span class="iter-status">${escapeHtml(status)}</span>`;
      frag.append(card);
    });
    list.replaceChildren(frag);
    [...list.querySelectorAll('.iter-card')].forEach((card) => {
      const prev = prevById.get(card.dataset.hid);
      if (prev && animate) {
        const rect = card.getBoundingClientRect();
        const dx = prev.rect.left - rect.left;
        const dy = prev.rect.top - rect.top;
        if (Math.abs(dx) > 0.5 || Math.abs(dy) > 0.5) {
          card.style.transition = 'none';
          card.style.transform = `translate(${dx}px, ${dy}px)`;
          requestAnimationFrame(() => {
            card.style.transition = 'transform 520ms ease';
            card.style.transform = '';
          });
        }
      }
      const scoreEl = card.querySelector('[data-score]');
      const prevScore = prev?.score;
      const newScore = Number(card.dataset.score);
      if (prevScore !== undefined && prevScore !== newScore && animate) {
        animateNumber(scoreEl, prevScore, newScore, 480);
      }
    });
    if (logEl && (isNewRound || !liveMode)) {
      const line = document.createElement('div');
      const parts = [];
      sorted.forEach((entry) => {
        const d = deltaFor(index, entry.hypothesis_id);
        if (d != null) parts.push(`${entry.hypothesis_id} 反馈 ${d > 0 ? '+' : ''}${numberValue(d, 2)}`);
        const v = verdictFor(index, entry.hypothesis_id);
        if (v) parts.push(`${entry.hypothesis_id} ${verdictLabel(v)}`);
      });
      line.textContent = `第 ${index + 1}/${rounds.length} 轮：${parts.length ? parts.join('、') : '初始排序（先验分）'}${index === 0 ? `（${entries.map((e) => `${e.hypothesis_id}=${numberValue(e.prior_score, 2)}`).join(' ')}）` : ''}`;
      logEl.append(line);
      logEl.scrollTop = logEl.scrollHeight;
    }
    if (label) label.textContent = `第 ${index + 1} / ${rounds.length} 轮`;
    panel.dataset.iterRound = String(index);
    roundIndex = index;
    if (index >= rounds.length - 1) stopPlay();
  };

  const goTo = (index) => renderRound(Math.max(0, Math.min(rounds.length - 1, index)), true);
  const togglePlay = () => {
    if (playing) { stopPlay(); return; }
    if (roundIndex >= rounds.length - 1) renderRound(0, false);
    playing = true;
    if (playButton) playButton.textContent = '⏸';
    timer = setInterval(() => {
      if (roundIndex >= rounds.length - 1) { stopPlay(); return; }
      renderRound(roundIndex + 1, true);
    }, speed);
  };

  panel.querySelectorAll('[data-iter-action]').forEach((button) => {
    button.addEventListener('click', () => {
      const action = button.dataset.iterAction;
      if (action === 'playpause') togglePlay();
      else if (action === 'restart') { stopPlay(); renderRound(0, false); }
      else if (action === 'prev') { stopPlay(); goTo(roundIndex - 1); }
      else if (action === 'next') { stopPlay(); goTo(roundIndex + 1); }
    });
  });
  if (speedSelect) {
    speedSelect.addEventListener('change', () => {
      speed = Number(speedSelect.value || 800);
      if (playing) {
        clearTimer();
        timer = setInterval(() => {
          if (roundIndex >= rounds.length - 1) { stopPlay(); return; }
          renderRound(roundIndex + 1, true);
        }, speed);
      }
    });
  }
  renderRound(liveMode ? rounds.length - 1 : 0, Boolean(externalPrev), externalPrev);
}

function renderPlanDetails(research) {
  const plan = research.researchPlan;
  return `<div class="stage-glance">
      ${infoRow('选定方案', research.selectedSolutionLabel)}
      ${infoRow('基线', research.baselines.length ? research.baselines.map(textValue).join('、') : 'Persistence')}
      ${infoRow('模型', research.modelFamily)}
      ${infoRow('执行门禁', research.leakageStatus, research.leakageStatus.includes('通过') ? 'ok' : '')}
    </div>
    <details class="research-detail"><summary>实验方案详情</summary><div class="research-detail-body">
      ${infoRow('研究目标', plan.objective || research.problem.research_goal || research.question)}
      ${infoRow('选定假设', plan.hypothesis_statement || research.selectedHypothesis)}
      ${infoRow('选择理由', plan.selection_reason || '按预声明综合评分确定')}
      ${infoRow('控制组', research.baselines.length ? research.baselines.map(textValue).join('、') : 'Persistence')}
      ${infoRow('门禁状态', research.leakageStatus)}
    </div></details>${research.selectionRationale ? renderSelectionRationale(research) : ''}`;
}

function renderSelectionRationale(research) {
  const detail = research.selectionDetail;
  const rationale = localizeSelectionRationale(research.selectionRationale);
  if (!rationale && !detail) return '';
  const rows = detail?.rows || [];
  const primary = detail?.primary_metric || research.metrics?.primary_metric || 'MAE';
  const lowerIsBetter = String(primary).toLowerCase() !== 'r2';
  const modelName = i18nModelName;
  const metrics = rows.length ? `<div class="selection-metrics">${rows.map((row) => {
    const selected = row.model === research.metrics?.selected_model;
    const lockedBest = row.model === research.metrics?.locked_test_best_model;
    const role = selected ? '协议选中' : lockedBest ? '锁定测试最优' : row.model === 'persistence' ? '参考基线' : '候选模型';
    const improvement = row.improvement_vs_persistence == null ? '—' : lowerIsBetter ? `${numberValue(row.improvement_vs_persistence, 2)}%` : numberValue(row.improvement_vs_persistence, 4);
    return `<article class="selection-metric-card ${selected || lockedBest ? 'is-highlighted' : ''}"><header><div><small>模型</small><h5>${escapeHtml(modelName(row.model))}</h5></div><span>${role}</span></header><dl><div><dt>验证集 ${escapeHtml(primary)}</dt><dd>${numberValue(row.validation_primary, 6)}</dd></div><div><dt>锁定测试集 ${escapeHtml(primary)}</dt><dd>${numberValue(row.locked_test_primary, 6)}</dd></div><div><dt>相对基线改进</dt><dd>${improvement}</dd></div></dl></article>`;
  }).join('')}</div>` : '';
  return `<section class="model-selection-rationale"><span class="overline">模型选择依据</span><h4>模型选择缘由</h4>${rationale ? `<p class="rationale-copy">${escapeHtml(rationale)}</p>` : ''}${metrics}</section>`;
}

function localizeSelectionRationale(value) {
  return localizeUiText(String(value || '')
    .replace(/Models and roles come from the frozen scientific design\.?\s*/gi, '模型与其分工均来自已冻结的科研设计。')
    .replace(/\blstm\b/gi, 'LSTM'));
}

function renderExecutionDetails(research) {
  const metrics = research.metrics;
  executionMetricsStore.set(research.runId, metrics);
  // 数值指标嵌在 locked/validation 明细里，合并后以锁定测试为准（大写键优先）。
  const displayMetrics = Object.assign({}, metrics.validation_metrics, metrics.locked_test_metrics, metrics);
  const semanticMetrics = semanticMetricEntries(displayMetrics);
  const keyMetrics = semanticMetrics.slice(0, 6);
  const environmentKeys = ['python_version', 'numpy_version', 'pandas_version', 'sklearn_version', 'torch_version', 'random_seed', 'experiment_adapter'];
  const environmentLabels = { python_version: 'Python 版本', numpy_version: 'NumPy 版本', pandas_version: 'Pandas 版本', sklearn_version: 'Scikit-learn 版本', torch_version: 'PyTorch 版本', random_seed: '随机种子', experiment_adapter: '实验适配器' };
  const liveElapsed = research.startedAt ? Math.max(0, (Date.now() - research.startedAt) / 1000) : null;
  const elapsed = research.elapsedSeconds === null ? (liveElapsed === null ? '运行中' : `${numberValue(liveElapsed, 0)} 秒（前端计时）`) : `${numberValue(research.elapsedSeconds, 1)} 秒`;
  const capabilities = research.capabilities?.experiment || research.capabilities?.capabilities || {};
  const capabilityEntries = Object.keys(capabilities).length ? [
    ['可用模型', (capabilities.models || []).join('、') || '—'],
    ['参考基线', capabilities.reference_model || '—'],
    ['可用指标', (capabilities.metrics || []).join('、') || '—'],
    ['支持操作', (capabilities.operations || []).join('、') || '—'],
    ['采样间隔', capabilities.sampling_interval_seconds != null ? `${capabilities.sampling_interval_seconds} 秒` : '—'],
    ['预测时域', capabilities.prediction_horizon_steps != null ? `${capabilities.prediction_horizon_steps} 步` : '—'],
    ['窗口步数', capabilities.window_steps != null ? `${capabilities.window_steps} 步` : '—'],
    ['数据切分', capabilities.splits?.policy || '—'],
    ['数据集', capabilities.dataset ? `${capabilities.dataset.id}（${capabilities.dataset.row_count ?? '—'} 行 · ${capabilities.dataset.real_industrial_data ? '真实工业数据' : '—'}）` : '—']
  ].map(([label, value]) => infoRow(label, value, 'ok')).join('') : '<p class="muted-copy">正在读取后端能力。</p>';
  const rows = research.modelRows || [];
  const selectedModel = metrics.selected_model;
  const lockedBestModel = metrics.locked_test_best_model;
  const persistence = rows.find((row) => String(row.model).toLowerCase() === 'persistence');
  const persistenceMae = persistence?.locked_test?.MAE ?? persistence?.locked_test?.mae_t_h;
  const selectedRow = rows.find((row) => row.model === selectedModel);
  const bestRow = rows.find((row) => row.model === lockedBestModel);
  const selectedValidationMae = selectedRow?.validation?.MAE ?? selectedRow?.validation?.mae_t_h;
  const bestLockedMae = bestRow?.locked_test?.MAE ?? bestRow?.locked_test?.mae_t_h;
  const improvement = Number(persistenceMae) && Number.isFinite(Number(bestLockedMae))
    ? (Number(persistenceMae) - Number(bestLockedMae)) / Number(persistenceMae) * 100
    : null;
  const modelName = i18nModelName;
  const modelNarrative = (row) => {
    const validationMae = row.validation?.MAE ?? row.validation?.mae_t_h;
    const lockedMae = row.locked_test?.MAE ?? row.locked_test?.mae_t_h;
    const gain = Number(persistenceMae) && Number.isFinite(Number(lockedMae)) && row.model !== 'persistence'
      ? (Number(persistenceMae) - Number(lockedMae)) / Number(persistenceMae) * 100
      : null;
    if (row.model === 'persistence') return `作为不学习参数的惯性参考模型，其锁定测试集 MAE 为 ${numberValue(lockedMae, 6)}，用于判断候选模型是否真正带来新增预测价值。`;
    const roles = [];
    if (row.model === selectedModel) roles.push('由验证集阶段按预声明协议选中');
    if (row.model === lockedBestModel) roles.push('在锁定测试集上取得最低 MAE');
    return `${roles.length ? roles.join('，') + '。' : ''}验证集 MAE 为 ${numberValue(validationMae, 6)}，锁定测试集 MAE 为 ${numberValue(lockedMae, 6)}${gain === null ? '。' : `，相对持久性基线模型降低 ${numberValue(gain, 2)}%。`}`;
  };
  return `<div class="stage-glance">
      ${infoRow('当前执行步骤', humanStage(research.rawStage))}
      ${infoRow('阶段耗时', elapsed)}
      ${infoRow('数据状态', research.datasetIsReal ? '真实锅炉历史数据' : '等待真实数据确认', research.datasetIsReal ? 'ok source' : '')}
      ${infoRow('样本数量', metrics.sample_count ? `${numberValue(metrics.sample_count, 0)} 条` : '等待数据产物')}
      ${infoRow('数据泄漏检查', research.leakageStatus, research.leakageStatus.includes('通过') ? 'ok' : '')}
      ${infoRow('已完成任务节点', `${(research.verification.executed_step_ids || []).length} 个`)}
    </div>
    ${keyMetrics.length ? `<section class="metric-key-grid" aria-label="关键指标">${keyMetrics.map((entry) => `<article><small>${escapeHtml(entry.label)}</small><strong>${escapeHtml(formatMetricDisplay(entry, 4))}</strong></article>`).join('')}</section>` : ''}
    ${rows.length ? `<section class="model-result-story"><span class="overline">模型发现</span><h4>模型比较结论</h4><p>按冻结实验协议，${escapeHtml(modelName(selectedModel))} 以验证集 MAE ${numberValue(selectedValidationMae, 6)} 被正式选中。独立锁定测试集中，${escapeHtml(modelName(lockedBestModel))} 的 MAE ${numberValue(bestLockedMae, 6)} 最低${improvement === null ? '。' : `，相对持久性基线模型改善 ${numberValue(improvement, 2)}%。`} ${selectedModel !== lockedBestModel ? '正式选择与锁定测试最优模型不一致，因此不能用锁定测试集回选模型，建议增加跨时间块复验。' : '正式选择与锁定测试结果一致，当前结论具有较好的协议一致性。'}</p></section><div class="model-story-grid">${rows.map((row) => `<article class="model-story-card ${row.model === lockedBestModel ? 'best' : ''}"><div><span>${row.model === selectedModel ? '协议选择' : row.model === lockedBestModel ? '锁定测试最优' : row.model === 'persistence' ? '参考基线' : '候选模型'}</span><h4>${escapeHtml(modelName(row.model))}</h4></div><p>${escapeHtml(modelNarrative(row))}</p></article>`).join('')}</div><details class="research-detail"><summary>查看技术指标明细</summary><div class="research-detail-body"><h4>验证集（仅用于模型选择）</h4>${renderModelMetricTable(rows, 'validation', selectedModel)}<h4>锁定测试集（仅用于泛化评价）</h4>${renderModelMetricTable(rows, 'locked_test', lockedBestModel)}</div></details>` : ''}
    <details class="research-detail"><summary>原始指标 <span>${semanticMetrics.length}</span></summary><div class="research-detail-body">
      <div class="metric-copy-row"><button class="secondary-button" type="button" data-action="copy-raw-metrics" data-run-id="${escapeHtml(research.runId)}">复制原始 JSON</button></div>
      <div class="raw-metrics">${semanticMetrics.length ? semanticMetrics.map((entry) => infoRow(entry.label, formatMetricDisplay(entry, 4))).join('') : '<p class="muted-copy">等待实验指标。</p>'}</div>
    </div></details>
    <details class="research-detail"><summary>运行环境</summary><div class="research-detail-body"><div class="environment-grid">${environmentKeys.map((key) => infoRow(environmentLabels[key], metrics[key])).join('')}</div></div></details>
    <details class="research-detail"><summary>研究产物 <span>${research.artifacts.length}</span></summary><div class="research-detail-body artifact-list">${research.artifacts.length ? research.artifacts.map((item) => `<code>${escapeHtml(textValue(item))}</code>`).join('') : '<p class="muted-copy">等待产物落盘。</p>'}</div></details>
    <details class="research-detail"><summary>系统能力详情</summary><div class="research-detail-body"><div class="environment-grid">${capabilityEntries}</div></div></details>`;
}

function renderModelMetricTable(rows, scope, highlightedModel) {
  return `<div class="metric-comparison-grid">${rows.map((row) => {
    const values = row[scope] || {};
    const highlighted = row.model === highlightedModel;
    return `<article class="metric-comparison-card ${highlighted ? 'is-highlighted' : ''}"><header><div><small>模型</small><strong>${escapeHtml(i18nModelName(row.model))}</strong></div>${highlighted ? '<span>当前最优</span>' : ''}</header><dl><div><dt>MAE（t/h）</dt><dd>${numberValue(values.MAE ?? values.mae_t_h, 6)}</dd></div><div><dt>RMSE（t/h）</dt><dd>${numberValue(values.RMSE ?? values.rmse_t_h, 6)}</dd></div><div><dt>MBE（t/h）</dt><dd>${numberValue(values.MBE ?? values.mbe_t_h, 6)}</dd></div></dl></article>`;
  }).join('')}</div>`;
}

function renderProblemDetails(research) {
  const problem = research.problem;
  const targetLabels = {
    steam_volumetric_flow: '蒸汽体积流量 V',
    main_steam_volumetric_flow: '主蒸汽体积流量 V'
  };
  const target = problem.target_variable || research.researchPlan.target || research.planBook.technical_details?.input_variables?.target;
  const horizonSteps = problem.required_horizon_steps || research.researchPlan.prediction_horizon_steps;
  const samplingSeconds = research.researchPlan.sampling_interval_seconds;
  const horizonMinutes = Number(horizonSteps) && Number(samplingSeconds)
    ? `${Number(horizonSteps) * Number(samplingSeconds) / 60} 分钟（h${horizonSteps}，采样间隔 ${samplingSeconds} 秒）`
    : problem.time_range || '按冻结实验契约执行';
  return `<div class="stage-glance">
    ${infoRow('研究目标', problem.research_goal || research.question)}
    ${infoRow('目标变量', targetLabels[target] || target)}
    ${infoRow('运行工况', problem.operating_condition || problem.operating_scenario || '未特别限定（全工况）')}
    ${infoRow('预测时域', research.metrics.task_time_semantics || horizonMinutes)}
  </div>`;
}

function renderEvaluationDetails(research) {
  const metrics = research.metrics;
  const primary = metrics.primary_metric || 'MAE';
  return `<div class="stage-glance">
    ${infoRow('假设结论', scientificText(research.scientificStatus))}
    ${infoRow(`候选模型 ${primary}`, numberValue(metrics.selected_model_primary))}
    ${infoRow(`Persistence ${primary}`, numberValue(metrics.persistence_primary))}
    ${infoRow('工程门禁', deploymentText(research.deploymentStatus))}
  </div>${renderSelectionRationale(research)}${renderIterationFeedback(research)}`;
}

function renderReportDetails(research) {
  const downloads = research.artifacts.filter((item) => item.download_url);
  return `<div class="stage-glance">
    ${infoRow('报告状态', research.completed ? '已生成' : research.currentStage >= 5 ? '正在生成' : '等待实验完成')}
    ${infoRow('研究产物', `${research.artifacts.length} 个`)}
  </div>${downloads.length ? `<div class="message-actions">${downloads.map((item) => `<button class="secondary-button" type="button" data-action="download-artifact" data-url="${escapeHtml(item.download_url)}">下载 ${escapeHtml(item.name)}</button>`).join('')}</div>` : ''}${research.completed ? `<button class="research-button stage-report-button" type="button" data-action="open-report" data-run-id="${escapeHtml(research.runId)}">阅读《科研假设与研究计划》</button>` : ''}`;
}

function directionLabel(current, recommended) {
  const currentNumber = Number(current);
  const recommendedNumber = Number(recommended);
  if (!Number.isFinite(currentNumber) || !Number.isFinite(recommendedNumber)) return '—';
  const delta = recommendedNumber - currentNumber;
  if (Math.abs(delta) < 1e-9) return '→ 保持';
  return delta > 0 ? '↑ 增大' : '↓ 减小';
}

function renderControlPlan(research) {
  const control = research.control;
  const current = control.current || {};
  const recommended = control.recommended || {};
  const currentValues = current.values || {};
  const ranges = recommended.ranges || {};
  const rangeEntries = Object.entries(ranges);
  return `<div class="stage-glance">
    ${infoRow('实验类型', control.experiment_type)}
    ${infoRow('当前蒸汽体积量 V', current.volume === undefined || current.volume === null ? '—' : `${numberValue(current.volume, 4)}`)}
    ${infoRow('推荐方案变量', `${rangeEntries.length} 个`)}
  </div>
  <details class="research-detail" open><summary>当前工况卡片</summary><div class="research-detail-body">
    ${Object.entries(currentValues).length ? `<div class="current-operating-grid">${Object.entries(currentValues).map(([name, value]) => infoRow(name, numberValue(value, 4), 'current-operating-item')).join('')}${current.volume === undefined || current.volume === null ? '' : infoRow('当前蒸汽体积量 V', numberValue(current.volume, 4), 'current-operating-item current-volume ok')}</div>` : '<p class="muted-copy">当前工况数据尚未生成。</p>'}
  </div></details>
  <details class="research-detail" open><summary>推荐方案卡片</summary><div class="research-detail-body">
    ${rangeEntries.length ? `<div class="control-recommendation-grid">${rangeEntries.map(([name, item]) => { const direction = directionLabel(item.current, item.recommended); const directionClass = direction.startsWith('↑') ? 'increase' : direction.startsWith('↓') ? 'decrease' : 'keep'; return `<article class="control-recommendation-card"><header><span>${escapeHtml(name)}</span><b class="${directionClass}">${escapeHtml(direction)}</b></header><div><small>当前值</small><strong>${numberValue(item.current, 4)}</strong></div><div><small>建议范围</small><strong>${numberValue(item.minimum, 4)} ～ ${numberValue(item.maximum, 4)}</strong></div><footer><span>推荐值</span><strong>${numberValue(item.recommended, 4)}</strong></footer></article>`; }).join('')}</div>` : '<p class="muted-copy">推荐调参范围尚未生成。</p>'}
  </div></details>
  <details class="research-detail" open><summary>约束卡片</summary><div class="research-detail-body">
    ${(control.constraints || []).length ? control.constraints.map((item) => infoRow('约束', constraintText(item), 'ok')).join('') : '<p class="muted-copy">无显式约束。</p>'}
  </div></details>`;
}

function renderControlExecution(research) {
  const control = research.control;
  const results = control.results || {};
  const unity = control.unity || {};
  const status = unity.status || 'none';
  const statusLabel = unityStatusLabels[status] || status || '未知';
  const verdictText = unitySecondVerdict(unity.second_verdict) || unity.second_verdict || '等待回传';
  const stepOrder = { none: -1, payload_generated: 0, sent: 1, received: 2, executed: 3, returned: 4 };
  const currentStep = stepOrder[status] ?? -1;
  const steps = ['指令生成', '已推送', '已接收', '已执行', '已回传'];
  const stepItems = steps.map((label, index) => `<span class="unity-step ${index < currentStep ? 'done' : index === currentStep ? 'active' : ''}"><i></i>${label}</span>`).join('');
  let guidance = '';
  if (status === 'none' || status === 'payload_generated') {
    guidance = '控制指令已由科研链路生成，等待推送到 Unity。打开「数字孪生」页面加载场景后，指令会自动推送、接收并回执；Unity 回传实际 V 后生成第二层裁决。';
  } else if (status === 'sent' || status === 'received' || status === 'executed') {
    guidance = `Unity 闭环进行中：等待${status === 'sent' ? ' Unity 确认接收' : status === 'received' ? ' Unity 执行调整' : ' Unity 回传实际 V'}。`;
  }
  const timeRows = [];
  if (unity.pushed_at) timeRows.push(infoRow('推送时间', formatTime(unity.pushed_at)));
  if (unity.received_at) timeRows.push(infoRow('接收时间', formatTime(unity.received_at)));
  if (unity.executed_at) timeRows.push(infoRow('执行时间', formatTime(unity.executed_at)));
  if (unity.returned_at) timeRows.push(infoRow('回传时间', formatTime(unity.returned_at)));
  const returned = unity.actual_volume !== undefined && unity.actual_volume !== null;
  const returnRows = returned ? [
    infoRow('实际 V', numberValue(unity.actual_volume, 4), 'ok'),
    infoRow('实际提升', `${numberValue(unity.actual_rise_pct * 100, 2)}%`),
    infoRow('与预测偏差', `${numberValue(unity.deviation_pct, 2)}%`),
    infoRow('Unity 闭环裁决', verdictText, 'ok')
  ].join('') : '<p class="muted-copy">实际 V、实际提升、与预测偏差、Unity 闭环裁决将在 Unity 回传后展示。</p>';
  return `<div class="stage-glance">
    ${infoRow('HGB 验证 MAE', numberValue(results.validation_mae, 6))}
    ${infoRow('当前 V', numberValue(results.current_volume, 4))}
    ${infoRow('目标 V', numberValue(results.target_volume, 4))}
    ${infoRow('预测 V', numberValue(results.predicted_volume, 4))}
    ${infoRow('预测提升', results.predicted_rise === undefined || results.predicted_rise === null ? '—' : `${numberValue(results.predicted_rise * 100, 2)}%`)}
    ${infoRow('可行候选', results.feasible_candidates === undefined || results.feasible_candidates === null ? '—' : `${numberValue(results.feasible_candidates, 0)} 个`)}
    ${infoRow('推荐压力', results.pressure_max_mpa === undefined || results.pressure_max_mpa === null ? '—' : `${numberValue(results.pressure_max_mpa, 2)} MPa`)}
  </div>
  <details class="research-detail" open><summary>Unity 推送状态</summary><div class="research-detail-body">
    ${infoRow('指令状态', statusLabel, unity.status === 'returned' ? 'ok' : '')}
    <div class="unity-steps">${stepItems}</div>
    ${guidance ? `<p class="unity-loop-hint">${guidance}</p>` : ''}
    ${timeRows.join('')}
    ${returnRows}
  </div></details>`;
}

function renderControlConclusion(research) {
  const control = research.control;
  const conclusion = control.conclusion || {};
  const verdictText = unitySecondVerdict(conclusion.verdict) || conclusion.verdict || '未裁决';
  const secondText = unitySecondVerdict(conclusion.second_verdict) || conclusion.second_verdict || '等待回传';
  return `<div class="stage-glance">
    ${infoRow('科学裁决', verdictText)}
    ${infoRow('结论范围', conclusion.scope ? controlConclusionScope(conclusion.scope) : '—')}
    ${infoRow('Unity 闭环', conclusion.unity_verified ? '已完成' : '等待 Unity 回传')}
    ${infoRow('第二层裁决', secondText)}
  </div>
  <details class="research-detail" open><summary>裁决依据</summary><div class="research-detail-body">
    ${infoRow('裁决说明', conclusion.rationale || '等待裁决')}
  </div></details>
  <div class="detail-block"><h3>说明</h3><p>当前结论来自 HGB 小模型软测验证；Unity 实际干预回传后才会形成第二层闭环裁决，不能把“JSON 已生成”当作真实干预已完成。</p></div>`;
}

function renderStageBody(stage, research) {
  if (stage.status === 'waiting') return '<p class="waiting-copy">等待前序阶段完成后自动展开。</p>';
  if (stage.id === 'problem') return renderProblemDetails(research);
  if (stage.id === 'evidence') return renderEvidenceDetails(research);
  if (stage.id === 'plan' && research.control) return renderControlPlan(research);
  if (stage.id === 'plan') return renderPlanDetails(research);
  if (stage.id === 'execution' && research.control) return renderControlExecution(research);
  if (stage.id === 'execution') return renderExecutionDetails(research);
  if (stage.id === 'evaluation' && research.control) return renderControlConclusion(research);
  if (stage.id === 'evaluation') return renderEvaluationDetails(research);
  return renderReportDetails(research);
}

export function renderProcessIndex(container, research, pendingLabel = '') {
  if (!container) return;
  container.hidden = false;
  const stages = research?.stages || ['问题拆解', '证据与假设', '实验方案', '实验执行', '科学评价', '科研报告'].map((title, index) => ({ id: ['problem', 'evidence', 'plan', 'execution', 'evaluation', 'report'][index], title, status: index === 0 ? 'active' : 'waiting' }));
  const runAttribute = research?.runId ? ` data-run-id="${escapeHtml(research.runId)}"` : '';
  const percent = Math.max(0, Math.min(100, Number(research?.progressPercent) || 0));
  container.innerHTML = `<div class="process-index-head"><span>研究过程</span><strong>${escapeHtml(research ? humanStage(research.rawStage) : pendingLabel || '正在理解问题')}</strong>${research ? `<div class="process-mini-progress"><i style="width:${percent}%"></i></div><small>${percent}% · 已完成 ${stages.filter((stage) => stage.status === 'completed').length}/${stages.length} 个阶段</small>` : ''}</div><nav>${stages.map((stage, index) => `<button type="button" data-action="focus-stage" data-stage-id="${stage.id}"${runAttribute} class="${stage.status}"><span>${index + 1}</span><div><strong>${stage.title}</strong><small>${stage.status === 'completed' ? '已完成' : stage.status === 'failed' ? '执行失败' : stage.status === 'active' ? '当前阶段' : '等待中'}</small></div></button>`).join('')}</nav>${research?.runId ? `<button class="process-run-link" type="button" data-action="return-active-run" data-run-id="${escapeHtml(research.runId)}">返回当前研究</button>` : ''}`;
}

export function createResearchElement(container, research) {
  const element = document.createElement('article');
  element.className = 'message research-run';
  element.dataset.runId = research.runId;
  container.append(element);
  updateResearchElement(element, research);
  return element;
}

function detailStateKey(detail) {
  const stageId = detail.closest('.stage')?.dataset.stage || 'root';
  const summary = detail.querySelector('summary')?.textContent.replace(/\s+/g, ' ').trim() || '';
  const parent = detail.closest('.stage') || detail.closest('.research-run');
  const details = [...parent.querySelectorAll('details')];
  const sameSummaryIndex = details.slice(0, details.indexOf(detail))
    .filter((item) => (item.querySelector('summary')?.textContent.replace(/\s+/g, ' ').trim() || '') === summary).length;
  return `${stageId}:${summary}:${sameSummaryIndex}`;
}

export function updateResearchElement(element, research) {
  const previousStage = Number(element.dataset.currentStage ?? -1);
  const firstRender = element.dataset.rendered !== 'true';
  const detailStates = new Map([...element.querySelectorAll('details')].map((detail) => [detailStateKey(detail), detail.open]));
  const percent = Math.max(0, Math.min(100, Number(research.progressPercent) || 0));
  const completedStages = research.stages.filter((stage) => stage.status === 'completed').length;
  element.innerHTML = `
    <div class="message-meta">研究任务 · ${escapeHtml(research.runId || '正在创建')}</div>
    <div class="research-question">${escapeHtml(research.question || '正在读取研究问题')}</div>
    <section class="research-progress-summary ${research.failed ? 'failed' : research.completed ? 'completed' : ''}"><div><span>${research.failed ? '任务已停止' : research.completed ? '研究已完成' : '真实研究正在执行'}</span><strong>${percent}%</strong></div><div class="research-progress-track"><i style="width:${percent}%"></i></div><p>${research.failed ? `失败阶段：${escapeHtml(research.stages.find((stage) => stage.status === 'failed')?.title || humanStage(research.rawStage))}` : `当前阶段：${escapeHtml(research.stages[research.currentStage]?.title || humanStage(research.rawStage))}`} · 已完成 ${completedStages}/${research.stages.length} 个阶段</p>${research.failed && research.message ? `<div class="research-error-reason"><strong>后端失败原因</strong><code>${escapeHtml(research.message)}</code></div>` : ''}</section>
    <div class="stage-list">
      ${research.stages.map((stage, index) => `
        <section class="stage ${stage.status} ${firstRender || index > previousStage ? 'stage-reveal' : ''}" data-stage="${stage.id}" style="--reveal-delay:${firstRender ? index * 100 : 0}ms">
          <div class="stage-marker">${stage.status === 'completed' ? icon('check-circle') : stage.status === 'waiting' ? icon('circle') : ''}</div>
          <div class="stage-head"><h3>${index + 1}. ${stage.title}</h3><span class="stage-state ${stage.status === 'active' ? 'live' : ''}">${stage.status === 'completed' ? '已完成' : stage.status === 'failed' ? '执行失败' : stage.status === 'active' ? escapeHtml(humanStage(research.rawStage)) : '等待中'}</span></div>
          <p class="stage-summary">${escapeHtml(localizeUiText(stage.summary))}</p>
          ${renderStageBody(stage, research)}
        </section>`).join('')}
    </div>
    ${research.needsHumanReview ? renderHumanReview(research) : ''}
    ${research.completed ? renderVerdict(research) : ''}
    ${research.completed ? renderKnowledgeGrowthSlot(research) : ''}
    ${research.failed ? `<div class="detail-block"><h3>研究执行失败</h3><p>${escapeHtml(research.message || '请查看后端运行日志。')}</p></div>` : ''}`;
  element.querySelectorAll('details').forEach((detail) => {
    const open = detailStates.get(detailStateKey(detail));
    if (open !== undefined) detail.open = open;
  });
  element.dataset.currentStage = String(research.currentStage);
  element.dataset.rendered = 'true';
}

function renderHumanReview(research) {
  const candidates = research.hypothesisCandidates.slice(0, 3);
  return `<div class="detail-block"><h3>需要人工选择假设</h3><p>当前候选假设无法自动决胜。请选择一个假设继续实验。</p><div class="message-actions">${candidates.map((item) => `<button class="secondary-button" type="button" data-action="select-hypothesis" data-run-id="${escapeHtml(research.runId)}" data-hypothesis-id="${escapeHtml(item.hypothesis_id || item.id)}">${escapeHtml(item.title || item.hypothesis_id || item.id)}</button>`).join('')}</div></div>`;
}

function renderVerdict(research) {
  return `<div class="verdict-strip"><div class="verdict"><span>执行状态</span><strong>实验已完成</strong></div><div class="verdict"><span>科学状态</span><strong>${escapeHtml(scientificText(research.scientificStatus))}</strong></div><div class="verdict"><span>工程状态</span><strong>${escapeHtml(deploymentText(research.deploymentStatus))}</strong></div></div><div class="message-actions"><button class="secondary-button" type="button" data-action="view-process">查看研究过程</button><button class="research-button" type="button" data-action="open-report" data-run-id="${escapeHtml(research.runId)}">阅读《科研假设与研究计划》</button></div>`;
}

function renderKnowledgeGrowthSlot(research) {
  return `<section class="kg-growth" data-kg-growth data-run-id="${escapeHtml(research.runId)}" aria-label="知识图谱增长">
    <div class="kg-growth-head">
      <div><span class="overline">科研知识演化</span><h3>知识图谱增长</h3><p data-kg-growth-summary>正在读取当前知识图谱与本次实验新增节点……</p></div>
      <div class="kg-growth-controls" role="group" aria-label="知识图谱视图">
        <button class="secondary-button is-active" type="button" data-kg-growth-mode="diff" aria-pressed="true">增量对比</button>
        <button class="secondary-button" type="button" data-kg-growth-mode="classic" aria-pressed="false">经典图谱</button>
        <button class="secondary-button" type="button" data-kg-growth-replay>重播生长</button>
      </div>
    </div>
    <div class="kg-growth-body" data-kg-growth-body><div class="feature-empty"><h3>正在构建知识图谱</h3><p>读取真实研究历史和科学裁决。</p></div></div>
  </section>`;
}

export function renderReport(container, report) {
  if (report?.metadata?.schema_version === 'boilermind.scientific_hypothesis_research_plan.v2') {
    return renderScientificPlan(container, report);
  }
  container.innerHTML = `<article class="report-document"><h1>科研假设与研究计划</h1><p>后端返回了旧版报告格式，请重新打开报告以获取 v2 版本。</p><pre>${escapeHtml(JSON.stringify(report, null, 2))}</pre></article>`;
  return {};
}

function renderScientificPlan(container, report) {
  const results = report.results || {};
  const rationale = report.rationale || {};
  const problem = report.problem_statement || {};
  const rows = results.model_comparison_rows || [];
  const runId = report.metadata?.run_id || report.run_id || '';
  const abstractText = report.paper_abstract?.rendered_text || report.paper_abstract?.polished_text || '';
  const downloadButtons = ['科研假设与研究计划（PDF）', '科研假设与研究计划（Word）', '科研假设与研究计划（Markdown）', '科研假设与研究计划（JSON）'].map((label, index) => {
    const artifactId = ['scientific_plan_pdf', 'scientific_plan_word', 'scientific_plan_markdown', 'scientific_plan_json'][index];
    return `<button class="secondary-button" type="button" data-action="download-artifact" data-url="/api/v1/research-runs/${encodeURIComponent(runId)}/artifacts/${artifactId}/download">下载 ${label}</button>`;
  }).join('');
  container.innerHTML = `<article class="report-document">
    ${runId ? `<div class="message-actions">${downloadButtons}</div>` : ''}
    <h1>${escapeHtml(report.paper_title || '科研假设与研究计划')}</h1>
    <div class="verdict-strip"><div class="verdict"><span>协议选定模型</span><strong>${escapeHtml(i18nModelName(results.protocol_selected_model))}</strong></div><div class="verdict"><span>锁定测试最优</span><strong>${escapeHtml(i18nModelName(results.locked_test_best_model))}</strong></div><div class="verdict"><span>科学裁决</span><strong>${escapeHtml(verdictLabel(results.scientific_verdict))}</strong></div></div>
    <h2>研究摘要</h2><p id="reportConclusion">${escapeHtml(abstractText)}</p>
    <h2>科研问题</h2><p>${escapeHtml(problem.original_question || '')}</p>
    <h2>主假设</h2><p>${escapeHtml(rationale.hypothesis_statement || '')}</p>
    <h2>竞争子假设与实验规划</h2>${(rationale.competing_hypotheses || []).map((item, index) => `<section class="detail-block"><h3>H${index + 1} · ${escapeHtml(item.title || i18nModelName(item.model))}</h3><p>${escapeHtml(item.statement || '')}</p><p><strong>预期观察：</strong>${escapeHtml(item.expected_observation || '')}</p>${item.experiment_plan ? `<details><summary>实验规划书</summary><div>${infoRow('实验目的', item.experiment_plan.objective)}${infoRow('实验设计', item.experiment_plan.design)}${infoRow('主指标', item.experiment_plan.primary_endpoint)}${infoRow('确认规则', item.experiment_plan.confirmation_rule)}${infoRow('证伪规则', item.experiment_plan.falsification_rule)}</div></details>` : ''}</section>`).join('')}
    <h2>模型比较</h2><div class="table-wrap"><table class="research-table"><thead><tr><th>模型</th><th>验证集</th><th>锁定测试集</th><th>相对 Persistence 改善</th></tr></thead><tbody>${rows.map((row) => `<tr><td>${escapeHtml(i18nModelName(row.model))}</td><td>${numberValue(row.validation_primary, 6)}</td><td>${numberValue(row.locked_test_primary, 6)}</td><td>${numberValue(row.locked_test_improvement_vs_baseline_pct, 2)}%</td></tr>`).join('')}</tbody></table></div>
    <h2>科学解释</h2><p>${escapeHtml(results.selection_interpretation || '')}</p>
    <h2>局限性</h2><ul>${(report.limitations || []).map((item) => `<li>${escapeHtml(item)}</li>`).join('')}</ul>
    <h2>参考文献</h2><ol>${(report.references || []).map((item) => `<li>${escapeHtml(item.formatted_citation || item.citation || item.title || '')}</li>`).join('')}</ol>
  </article>`;
  return { model_rows: rows };
}

