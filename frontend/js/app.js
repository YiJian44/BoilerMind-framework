import { CONFIG } from './config.js?v=20260826-kg23';
import { api } from './api-client.js?v=20260826-kg18';
import { state, newSession, setMode, rememberRun } from './state.js?v=20260826-kg4';
import { normalizeResearch } from './research-adapter.js?v=20260826-kg4';
import { renderHistoryList, renderHistoryTable, renderProcessIndex, appendUserMessage, appendLoadingMessage, appendAssistantProgress, appendResearchLaunch, renderAssistantMessage, renderAssistantError, createResearchElement, updateResearchElement, renderReport, executionMetricsStore, initIterationReplay, captureIterationState } from './views.js?v=20260829-localize3';
import { escapeHtml, semanticMetricEntries, formatMetricDisplay, rawMetricsJson, statusText } from './formatters.js?v=20260826-kg9';

const $ = (selector, root = document) => root.querySelector(selector);

function localizeReportText(value) {
  return String(value ?? '')
    .replaceAll('hgb_control_optimizer', 'HGB 控制优化器')
    .replaceAll('locked test 最优', '锁定测试集最优')
    .replaceAll('使用 locked test 回选', '使用锁定测试集回选')
    .replaceAll('locked test 不参与', '锁定测试集不参与')
    .replaceAll('locked-test', '锁定测试集')
    .replaceAll('locked test', '锁定测试集')
    .replaceAll('validation-only', '仅验证集')
    .replaceAll('validation', '验证集');
}
const elements = {
  shell: $('#appShell'), drawer: $('#conversationDrawer'), stream: $('#messageStream'), welcome: $('#welcomeState'), scroll: $('#conversationScroll'), composer: $('#composerForm'), input: $('#composerInput'), mode: $('#composerMode'), modeMenu: $('#modeMenu'), modeTrigger: $('#modeTrigger'), modeTriggerLabel: $('#modeTriggerLabel'), send: $('#sendButton'), attachmentList: $('#attachmentList'), fileInput: $('#fileInput'), historyList: $('#conversationList'), historyTable: $('#historyTable'), historyCount: $('#historyCount'), progressPill: $('#progressPill'), progressText: $('#progressPillText'), progressDot: $('.progress-dot'), drawerSystemText: $('#drawerSystemText'), drawerSystemDot: $('#drawerSystemDot'), reportReader: $('#reportReader'), newProgress: $('#newProgressButton'), unityStage: $('#unityStage'), processSidebar: $('#processSidebar')
};


function setDrawer(open) {
  const available = elements.shell.dataset.activeView !== 'history';
  const visible = open && available;
  elements.shell.classList.toggle('drawer-open', visible);
  elements.drawer.setAttribute('aria-hidden', String(!visible));
  document.querySelectorAll('[data-action="toggle-drawer"]').forEach((button) => button.setAttribute('aria-expanded', String(visible)));
}

function showView(name) {
  if (!document.querySelector(`[data-view-panel="${name}"]`)) name = 'chat';
  elements.shell.dataset.activeView = name;
  document.querySelectorAll('[data-action="toggle-drawer"]').forEach((button) => {
    button.disabled = name === 'history';
    button.setAttribute('aria-disabled', String(name === 'history'));
  });
  document.querySelectorAll('[data-view-panel]').forEach((panel) => panel.classList.toggle('is-active', panel.dataset.viewPanel === name));
  document.querySelectorAll('.rail-button[data-view]').forEach((button) => button.classList.toggle('is-active', button.dataset.view === name));
  const titles = { chat: '研究对话', unity: 'Unity 数字孪生', knowledge: '知识图谱', history: '历史任务', report: '科研报告' };
  $('#viewTitle').textContent = titles[name] || 'BoilerMind';
  if (location.hash !== `#/${name}`) history.replaceState(null, '', `${location.pathname}${location.search}#/${name}`);
  if (name === 'history') loadHistory();
  if (name === 'knowledge') loadKnowledgeGraph();
  setDrawer(false);
}

const knowledgeLabels = {
  supported: ['得到支持', '满足预声明确认标准，并保留实验边界与评价协议。'],
  partial: ['部分支持', '仅在部分指标或工况成立，保留边界条件和复验方向。'],
  insufficient: ['证据不足', '当前实验不足以形成稳定裁决，记录证据缺口。'],
  falsified: ['被证伪', '实验触发证伪标准，保留反例并推动替代假设演化。']
};

function knowledgeStatus(value) {
  const status = String(value || '').toLowerCase();
  if (status.includes('partial')) return 'partial';
  if (status.includes('fals')) return 'falsified';
  if (status.includes('insufficient') || status.includes('inconclusive') || status.includes('not_executed')) return 'insufficient';
  if (status.includes('support') || status.includes('validated')) return 'supported';
  return 'insufficient';
}

function fitSvgText(text, max) {
  const value = String(text || '').trim();
  return value.length > max ? `${value.slice(0, max - 1)}…` : value;
}

function wrapSvgText(text, max) {
  const value = String(text || '').replace(/\s+/g, ' ').trim();
  const lines = [];
  let rest = value;
  while (rest.length > max && lines.length < 2) {
    lines.push(rest.slice(0, max));
    rest = rest.slice(max);
  }
  lines.push(rest.slice(0, max));
  return lines;
}

function primaryMetricText(metrics) {
  return semanticMetricEntries(metrics).slice(0, 3).map((entry) => `${entry.label} ${formatMetricDisplay(entry, 3)}`).join(' · ');
}

function metricChips(metrics) {
  return semanticMetricEntries(metrics).slice(0, 6).map((entry) => `<span>${escapeHtml(entry.label)} ${escapeHtml(formatMetricDisplay(entry, 3))}</span>`).join('');
}

function isSuccessfulValidation(validatedNode, experimentNode) {
  if (!validatedNode || !experimentNode) return false;
  if (knowledgeStatus(validatedNode.validation_status) !== 'supported') return false;
  if (experimentNode.experiment_valid === false) return false;
  if (String(experimentNode.status || '').toLowerCase() !== 'completed') return false;
  if (knowledgeStatus(experimentNode.verdict) !== 'supported') return false;
  return true;
}

function computeForceLayout(nodeList, edgeList, width, height) {
  const nodeRadius = { hypothesis: 46, experiment: 32, validated: 38, mechanism: 26, scope: 26, metric: 26 };
  const nodePadding = 16;
  const ringOf = { hypothesis: 0, experiment: 1, validated: 2, mechanism: 3, scope: 3, metric: 3 };
  const positions = new Map();
  nodeList.forEach((node, index) => {
    const ring = ringOf[node.type] ?? 1;
    const ringNodes = nodeList.filter((n) => (ringOf[n.type] ?? 1) === ring);
    const ringIndex = ringNodes.findIndex((n) => n.id === node.id);
    const count = Math.max(ringNodes.length, 1);
    const angle = (ringIndex / count) * Math.PI * 2 + ring * 0.35;
    const radius = 110 + ring * 150 + (node.type === 'metric' ? -30 : 0);
    positions.set(node.id, {
      x: width / 2 + Math.cos(angle) * radius,
      y: height / 2 + Math.sin(angle) * radius * 0.9,
      vx: 0,
      vy: 0
    });
  });
  const iterations = nodeList.length > 36 ? 620 : 460;
  const kRep = 11000;
  const kSpring = 0.045;
  const restLength = 160;
  const kCenter = 0.012;
  const centerX = width / 2;
  const centerY = height / 2;
  for (let iter = 0; iter < iterations; iter++) {
    for (let i = 0; i < nodeList.length; i++) {
      for (let j = i + 1; j < nodeList.length; j++) {
        const a = positions.get(nodeList[i].id);
        const b = positions.get(nodeList[j].id);
        const dx = b.x - a.x;
        const dy = b.y - a.y;
        const d2 = dx * dx + dy * dy || 1;
        const d = Math.sqrt(d2);
        if (d > 340) continue;
        const force = kRep / d2;
        const fx = (dx / d) * force;
        const fy = (dy / d) * force;
        a.vx -= fx;
        a.vy -= fy;
        b.vx += fx;
        b.vy += fy;
      }
    }
    edgeList.forEach((edge) => {
      const a = positions.get(edge.source);
      const b = positions.get(edge.target);
      if (!a || !b) return;
      const dx = b.x - a.x;
      const dy = b.y - a.y;
      const d = Math.sqrt(dx * dx + dy * dy) || 1;
      const force = kSpring * (d - restLength);
      const fx = (dx / d) * force;
      const fy = (dy / d) * force;
      a.vx += fx;
      a.vy += fy;
      b.vx -= fx;
      b.vy -= fy;
    });
    for (const node of nodeList) {
      const p = positions.get(node.id);
      p.vx += (centerX - p.x) * kCenter;
      p.vy += (centerY - p.y) * kCenter;
      p.vx *= 0.86;
      p.vy *= 0.86;
      const radius = node.radius || nodeRadius[node.type] || 26;
      p.x = Math.max(radius + nodePadding, Math.min(width - radius - nodePadding, p.x + p.vx));
      p.y = Math.max(radius + nodePadding, Math.min(height - radius - nodePadding, p.y + p.vy));
    }
  }
  // 排斥力不足以保证零重叠；最终用圆形碰撞约束把相交节点推开。
  for (let pass = 0; pass < 480; pass++) {
    let hasOverlap = false;
    for (let i = 0; i < nodeList.length; i++) {
      for (let j = i + 1; j < nodeList.length; j++) {
        const a = positions.get(nodeList[i].id);
        const b = positions.get(nodeList[j].id);
        const minDistance = (nodeList[i].radius || nodeRadius[nodeList[i].type] || 26)
          + (nodeList[j].radius || nodeRadius[nodeList[j].type] || 26) + nodePadding;
        let dx = b.x - a.x;
        let dy = b.y - a.y;
        let distance = Math.hypot(dx, dy);
        if (distance >= minDistance) continue;
        hasOverlap = true;
        if (distance < 0.001) {
          dx = i < j ? 1 : -1;
          dy = 0;
          distance = 1;
        }
        const shift = (minDistance - distance) / 2;
        const ux = dx / distance;
        const uy = dy / distance;
        a.x -= ux * shift;
        a.y -= uy * shift;
        b.x += ux * shift;
        b.y += uy * shift;
        for (const [node, point] of [[nodeList[i], a], [nodeList[j], b]]) {
          const radius = node.radius || nodeRadius[node.type] || 26;
          point.x = Math.max(radius + nodePadding, Math.min(width - radius - nodePadding, point.x));
          point.y = Math.max(radius + nodePadding, Math.min(height - radius - nodePadding, point.y));
        }
      }
    }
    if (!hasOverlap) break;
  }
  return positions;
}
let knowledgeDetailNodes = new Map();
let knowledgeDetailContainer = null;

const KG_SEEN_KEY = 'boilermind.kg.seen.nodes.v1';

function loadSeenKgNodes() {
  try {
    return new Set(JSON.parse(localStorage.getItem(KG_SEEN_KEY) || '[]'));
  } catch {
    return new Set();
  }
}

function saveSeenKgNodes(ids) {
  try {
    localStorage.setItem(KG_SEEN_KEY, JSON.stringify([...ids]));
  } catch {
    // 存储不可用时仅跳过新增高亮，不影响图谱渲染。
  }
}

function hideKnowledgeDetail() {
  if (!knowledgeDetailContainer) return;
  const existing = knowledgeDetailContainer.querySelector('.kg-detail-card');
  if (existing) existing.remove();
}

function showKnowledgeDetail(nodeId) {
  hideKnowledgeDetail();
  const node = knowledgeDetailNodes.get(nodeId);
  if (!node) return;
  const typeLabels = { Hypothesis: '科学假设', ExperimentResult: '实验验证', ValidatedHypothesis: '已验证假设', mechanism: '机理链', scope: '验证范围', metric: '关键指标' };
  const detailKinds = new Set(['mechanism', 'scope', 'metric']);
  const isDetail = detailKinds.has(node.type);
  const statusValue = isDetail ? '' : (node.type === 'ValidatedHypothesis' ? node.validation_status : (node.type === 'Hypothesis' ? node.status : node.verdict));
  const statusText = isDetail ? '验证细节' : knowledgeLabels[knowledgeStatus(statusValue)][0];
  const content = node.content || node.statement || '';
  const sections = [];
  if (content) sections.push(`<p>${escapeHtml(content)}</p>`);
  if (node.mechanism_chain) sections.push(`<dl><dt>机理链</dt><dd>${escapeHtml(node.mechanism_chain)}</dd></dl>`);
  if (node.type === 'ExperimentResult') {
    sections.push(`<dl><dt>实验有效性</dt><dd>${node.experiment_valid === false ? '无效（已被排除）' : '有效'} · 状态 ${escapeHtml(statusText(node.status))}</dd><dt>关键指标</dt><dd><div class="kg-detail-metrics">${metricChips(node.metrics)}</div></dd></dl>`);
  }
  if (node.type === 'ValidatedHypothesis') {
    sections.push(`<dl><dt>关键指标</dt><dd><div class="kg-detail-metrics">${metricChips(node.metrics)}</div></dd></dl>`);
  }
  const card = document.createElement('div');
  card.className = 'kg-detail-card';
  card.innerHTML = `<button class="kg-detail-close" type="button" aria-label="关闭详情">×</button>
    <span class="kg-detail-type">${typeLabels[node.type] || node.type}</span>
    <h3>${escapeHtml(statusText)}</h3>
    <div class="kg-detail-id">${escapeHtml(node.id)}</div>
    ${sections.join('')}`;
  card.querySelector('.kg-detail-close').addEventListener('click', hideKnowledgeDetail);
  if (knowledgeDetailContainer) knowledgeDetailContainer.append(card);
}

function renderKnowledgeGraph(graph, options = {}) {
  const container = options.container || $('#knowledgeGraph');
  const standalone = options.standalone === true;
  const diffMode = options.newNodeIds instanceof Set;
  const nodes = graph.nodes || [];
  const edges = graph.edges || [];
  const hypotheses = nodes.filter((node) => node.type === 'Hypothesis');
  const experiments = nodes.filter((node) => node.type === 'ExperimentResult');
  const validated = nodes.filter((node) => node.type === 'ValidatedHypothesis');
  const nodesById = new Map(nodes.map((node) => [node.id, node]));

  const counts = { supported: 0, partial: 0, insufficient: 0, falsified: 0 };
  const statusOf = (node) => knowledgeStatus(
    node.type === 'ValidatedHypothesis' ? node.validation_status : node.status
  );
  [...hypotheses, ...validated].forEach((node) => { counts[statusOf(node)] += 1; });
  if (!standalone) {
    $('#knowledgeStatusGrid').innerHTML = Object.entries(knowledgeLabels).map(([status, [title, copy]]) => `<article class="${status}"><span>${counts[status]}</span><div><h3>${title}</h3><p>${copy}</p></div></article>`).join('');
  }

  const validatedBy = new Map();
  const generatesBy = new Map();
  edges.forEach((edge) => {
    if (edge.type === 'validated_by') {
      if (!validatedBy.has(edge.source)) validatedBy.set(edge.source, []);
      validatedBy.get(edge.source).push(edge.target);
    }
    if (edge.type === 'generates') generatesBy.set(edge.target, edge.source);
  });
  const hypothesisOfExperiment = new Map();
  validatedBy.forEach((experimentIds, hypothesisId) => {
    experimentIds.forEach((experimentId) => hypothesisOfExperiment.set(experimentId, hypothesisId));
  });

  // 只呈现成功验证链：已验证假设(supported) + 实验(experiment_valid && completed && verdict=support)
  const rows = [];
  validated.forEach((vh) => {
    const experiment = nodesById.get(vh.experiment_id || generatesBy.get(vh.id));
    if (!isSuccessfulValidation(vh, experiment)) return;
    const hypothesis = nodesById.get(hypothesisOfExperiment.get(experiment.id) || vh.hypothesis_id);
    if (!hypothesis || knowledgeStatus(hypothesis.status) === 'falsified') return;
    rows.push({ hypothesis, experiment, validated: vh });
  });

  // 强制补充调用方指定的本次实验关联节点（即使不在成功验证链内，重复实验也能显示）。
  if (options.extraNodeIds instanceof Set && options.extraNodeIds.size) {
    const presentRowIds = new Set(rows.flatMap((row) => [row.hypothesis.id, row.experiment.id, row.validated.id]));
    const seenHypothesis = new Set(rows.map((row) => row.hypothesis.id));
    const seenExperiment = new Set(rows.map((row) => row.experiment.id));
    nodes.forEach((node) => {
      if (presentRowIds.has(node.id) || !options.extraNodeIds.has(node.id)) return;
      if (node.type === 'Hypothesis' && !seenHypothesis.has(node.id)) {
        rows.push({ hypothesis: node, experiment: null, validated: null });
        seenHypothesis.add(node.id);
      } else if (node.type === 'ExperimentResult' && !seenExperiment.has(node.id)) {
        const hypothesisId = hypothesisOfExperiment.get(node.id) || node.hypothesis_id;
        rows.push({ hypothesis: hypothesisId ? nodesById.get(hypothesisId) || null : null, experiment: node, validated: null });
        seenExperiment.add(node.id);
      }
    });
  }

  const shownValidatedCount = new Set(rows.map((row) => row.validated?.id).filter(Boolean)).size;
  if (!standalone) {
    $('#knowledgeSummary').textContent = `已沉淀 ${hypotheses.length} 条假设 · ${shownValidatedCount} 条已验证假设 · ${rows.length} 次成功实验`;
  }

  if (!rows.length) {
    container.innerHTML = '<div class="feature-empty"><h3>尚无可沉淀的已验证假设</h3><p>完成一次真实研究后，假设、证据、实验和科学裁决会自动出现在这里。</p></div>';
    return;
  }

  // 新增节点高亮：优先使用调用方传入的本次运行新增集合；否则与上次查看快照对比。
  const currentKgIds = new Set();
  rows.forEach((row) => {
    if (row.hypothesis) currentKgIds.add(row.hypothesis.id);
    if (row.experiment) currentKgIds.add(row.experiment.id);
    if (row.validated) currentKgIds.add(row.validated.id);
  });
  let newKgNodes;
  if (options.newNodeIds instanceof Set) {
    newKgNodes = options.newNodeIds;
  } else {
    const seenKgNodes = loadSeenKgNodes();
    newKgNodes = seenKgNodes.size
      ? new Set([...currentKgIds].filter((id) => !seenKgNodes.has(id)))
      : new Set();
    if (options.saveSeen !== false) saveSeenKgNodes(new Set([...seenKgNodes, ...currentKgIds]));
  }

  rows.sort((a, b) => String(a.hypothesis?.id || a.experiment?.id || '').localeCompare(String(b.hypothesis?.id || b.experiment?.id || '')));

  // —— 网状知识图谱：节点与关系边（仅成功验证链，已验证假设拆解为细节节点）——
  const graphNodes = [];
  const graphEdges = [];
  const detailNodesById = new Map(nodesById);
  const detailSeen = new Map();
  const nodeRadius = { hypothesis: 46, experiment: 32, validated: 38, mechanism: 26, scope: 26, metric: 26 };
  const pushNode = (node) => {
    graphNodes.push(node);
    if (!detailNodesById.has(node.id)) {
      detailNodesById.set(node.id, { id: node.id, type: node.type, content: node.detail || node.content || '', metrics: node.metrics });
    }
  };
  const pushEdge = (source, target, rel) => graphEdges.push({ source, target, rel });
  const pushDetail = (kind, content, label) => {
    const key = `${kind}::${content}`;
    let id = detailSeen.get(key);
    if (!id) {
      id = `detail-${kind}-${graphNodes.length}`;
      detailSeen.set(key, id);
      pushNode({ id, type: kind, label, radius: nodeRadius[kind], content, detail: content });
    }
    return id;
  };

  rows.forEach((row) => {
    if (row.hypothesis) {
      pushNode({ id: row.hypothesis.id, type: 'hypothesis', label: '科学假设', sub: fitSvgText(row.hypothesis.id, 12), radius: nodeRadius.hypothesis, content: row.hypothesis.content, isNew: newKgNodes.has(row.hypothesis.id) });
    }
    if (row.experiment) {
      pushNode({ id: row.experiment.id, type: 'experiment', label: '实验验证', sub: fitSvgText(row.experiment.id, 14), radius: nodeRadius.experiment, content: row.experiment.content || '', metrics: row.experiment.metrics, isNew: newKgNodes.has(row.experiment.id) });
    }
    if (row.hypothesis && row.experiment) {
      pushEdge(row.hypothesis.id, row.experiment.id, 'validated_by');
    }
    if (!row.validated) return;
    pushNode({ id: row.validated.id, type: 'validated', label: '已验证假设', sub: knowledgeLabels[knowledgeStatus(row.validated.validation_status)][0], radius: nodeRadius.validated, content: row.validated.content, mechanism_chain: row.validated.mechanism_chain, metrics: row.validated.metrics, isNew: newKgNodes.has(row.validated.id) });
    pushEdge(row.experiment.id, row.validated.id, 'generates');

    const mechanism = row.validated.mechanism_chain || row.validated.content;
    const scope = row.validated.applicable_scope || '';
    if (mechanism) {
      const id = pushDetail('mechanism', mechanism, '机理链');
      pushEdge(row.validated.id, id, 'has_mechanism');
    }
    if (scope && scope !== mechanism) {
      const id = pushDetail('scope', scope, '验证范围');
      pushEdge(row.validated.id, id, 'has_scope');
    }
    const metricText = primaryMetricText(row.validated.metrics || row.experiment.metrics);
    if (metricText) {
      const id = pushDetail('metric', metricText, '关键指标');
      pushEdge(row.validated.id, id, 'has_metric');
    }
  });

  const seenNodes = new Set();
  const uniqueNodes = graphNodes.filter((node) => {
    if (seenNodes.has(node.id)) return false;
    seenNodes.add(node.id);
    return true;
  });
  const seenEdges = new Set();
  const uniqueEdges = graphEdges.filter((edge) => {
    const key = `${edge.source}::${edge.target}::${edge.rel}`;
    if (seenEdges.has(key)) return false;
    seenEdges.add(key);
    return true;
  });

  const width = 1120;
  const height = Math.max(820, 820 + Math.max(0, uniqueNodes.length - 30) * 16);
  const positions = computeForceLayout(uniqueNodes, uniqueEdges, width, height);

  const relLabels = { validated_by: '验证', generates: '生成', has_mechanism: '机理', has_scope: '范围', has_metric: '指标' };
  const neighborMap = new Map();
  uniqueEdges.forEach((edge) => {
    if (!neighborMap.has(edge.source)) neighborMap.set(edge.source, new Set());
    if (!neighborMap.has(edge.target)) neighborMap.set(edge.target, new Set());
    neighborMap.get(edge.source).add(edge.target);
    neighborMap.get(edge.target).add(edge.source);
  });

  const edgeMarkup = uniqueEdges.map((edge, index) => {
    const a = positions.get(edge.source);
    const b = positions.get(edge.target);
    if (!a || !b) return '';
    const midX = (a.x + b.x) / 2;
    const midY = (a.y + b.y) / 2;
    const edgeIsNew = options.newEdgeKeys instanceof Set && options.newEdgeKeys.has(`${edge.source}::${edge.rel}::${edge.target}`);
    return `<g class="kg-link ${edgeIsNew ? 'is-new' : ''}" data-edge-index="${index}" data-source="${escapeHtml(edge.source)}" data-target="${escapeHtml(edge.target)}">
      <line x1="${a.x.toFixed(1)}" y1="${a.y.toFixed(1)}" x2="${b.x.toFixed(1)}" y2="${b.y.toFixed(1)}" class="kg-edge kg-edge-${escapeHtml(edge.rel)}"/>
      <text x="${midX.toFixed(1)}" y="${midY.toFixed(1)}" text-anchor="middle" class="kg-edge-label">${escapeHtml(relLabels[edge.rel] || edge.rel)}</text>
      <title>${escapeHtml(relLabels[edge.rel] || edge.rel)}：${escapeHtml(edge.source)} → ${escapeHtml(edge.target)}</title>
    </g>`;
  }).join('');

  const nodeMarkup = uniqueNodes.map((node) => {
    const point = positions.get(node.id);
    if (!point) return '';
    const metaLine = node.sub || '';
    return `<g class="kg-clickable ${node.isNew ? 'is-new' : ''}" data-node-id="${escapeHtml(node.id)}" data-type="${escapeHtml(node.type)}" data-radius="${node.radius}" transform="translate(${point.x.toFixed(1)},${point.y.toFixed(1)})">
      <circle r="${node.radius}" class="kg-node kg-node-${escapeHtml(node.type)} kg-node-circle"/>
      <text y="${node.radius > 30 ? -8 : -4}" text-anchor="middle" class="kg-node-title">${escapeHtml(node.label)}</text>
      ${metaLine ? `<text y="14" text-anchor="middle" class="kg-node-meta">${escapeHtml(fitSvgText(metaLine, 16))}</text>` : ''}
      <title>${escapeHtml(node.label)}${metaLine ? ` · ${escapeHtml(metaLine)}` : ''}${node.content ? `：${escapeHtml(node.content)}` : ''}</title>
    </g>`;
  }).join('');

  knowledgeDetailNodes = detailNodesById;
  knowledgeDetailContainer = container;
  container.classList.toggle('kg-diff-mode', diffMode);
  container.innerHTML = `<div class="literature-legend">
    <span><i class="kg-dot kg-node-hypothesis"></i>科学假设</span>
    <span><i class="kg-dot kg-node-experimentresult"></i>实验验证</span>
    <span><i class="kg-dot kg-node-validated"></i>已验证假设</span>
    <span><i class="kg-dot kg-node-mechanism"></i>机理链</span>
    <span><i class="kg-dot kg-node-scope"></i>验证范围</span>
    <span><i class="kg-dot kg-node-metric"></i>关键指标</span>
  </div>
  <div class="kg-link-legend">
    <span><i class="kg-link-dot kg-edge-validated_by"></i>验证</span>
    <span><i class="kg-link-dot kg-edge-generates"></i>生成</span>
    <span><i class="kg-link-dot kg-edge-has_mechanism"></i>机理</span>
    <span><i class="kg-link-dot kg-edge-has_scope"></i>范围</span>
    <span><i class="kg-link-dot kg-edge-has_metric"></i>指标</span>
    <span class="kg-legend-hint">${diffMode ? '新增节点依次显现 · 旧节点置灰 · 拖动节点，关系线实时跟随 · 点击查看详情' : '拖动节点，关系线实时跟随 · 点击查看详情'}</span>
  </div>
  <div class="kg-svg-scroll">
    <svg class="kg-svg" viewBox="0 0 ${width} ${height}" preserveAspectRatio="xMidYMid meet" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="科研假设演化图谱">
      ${edgeMarkup}${nodeMarkup}
    </svg>
  </div>`;
  const svg = $('svg', container);
  const graphScroll = $('.kg-svg-scroll', container);
  requestAnimationFrame(() => {
    if (graphScroll) graphScroll.scrollTop = Math.max(0, (graphScroll.scrollHeight - graphScroll.clientHeight) / 2);
  });
  const edgeElements = Array.from(svg.querySelectorAll('g.kg-link'));
  const clearGraphHighlight = () => {
    svg.classList.remove('has-hover');
    svg.querySelectorAll('.is-hover, .is-linked').forEach((el) => el.classList.remove('is-hover', 'is-linked'));
    edgeElements.forEach((el) => el.classList.remove('is-highlight'));
  };
  svg.addEventListener('mouseover', (event) => {
    const group = event.target.closest('g.kg-clickable');
    if (!group) return;
    svg.classList.add('has-hover');
    const id = group.getAttribute('data-node-id');
    group.classList.add('is-hover');
    (neighborMap.get(id) || []).forEach((neighborId) => {
      const neighbor = svg.querySelector(`g.kg-clickable[data-node-id="${CSS.escape(neighborId)}"]`);
      if (neighbor) neighbor.classList.add('is-linked');
    });
    edgeElements.forEach((el) => {
      if (el.dataset.source === id || el.dataset.target === id) el.classList.add('is-highlight');
    });
  });
  svg.addEventListener('mouseout', (event) => {
    const group = event.target.closest('g.kg-clickable');
    if (group && group.contains(event.relatedTarget)) return;
    clearGraphHighlight();
  });
  svg.addEventListener('click', (event) => {
    const target = event.target.closest ? event.target.closest('g[data-node-id]') : null;
    if (!target) { hideKnowledgeDetail(); return; }
    showKnowledgeDetail(target.getAttribute('data-node-id'));
  });
  let dragNodeId = null;
  let dragOffset = { dx: 0, dy: 0 };
  const nodePosition = (nodeId) => {
    const group = svg.querySelector(`g.kg-clickable[data-node-id="${CSS.escape(nodeId)}"]`);
    const match = /translate\(([-\d.]+),\s*([-\d.]+)\)/.exec(group?.getAttribute('transform') || '');
    return match ? { x: Number(match[1]), y: Number(match[2]) } : null;
  };
  const updateLinkedEdges = (nodeId) => {
    edgeElements.forEach((edge) => {
      if (edge.dataset.source !== nodeId && edge.dataset.target !== nodeId) return;
      const source = nodePosition(edge.dataset.source);
      const target = nodePosition(edge.dataset.target);
      if (!source || !target) return;
      const line = edge.querySelector('line');
      const label = edge.querySelector('text');
      line?.setAttribute('x1', source.x.toFixed(1));
      line?.setAttribute('y1', source.y.toFixed(1));
      line?.setAttribute('x2', target.x.toFixed(1));
      line?.setAttribute('y2', target.y.toFixed(1));
      label?.setAttribute('x', ((source.x + target.x) / 2).toFixed(1));
      label?.setAttribute('y', ((source.y + target.y) / 2).toFixed(1));
    });
  };
  const constrainNodePosition = (nodeId, x, y) => {
    const group = svg.querySelector(`g.kg-clickable[data-node-id="${CSS.escape(nodeId)}"]`);
    const radius = Number(group?.dataset.radius) || 26;
    const padding = 16;
    const point = {
      x: Math.max(radius + padding, Math.min(width - radius - padding, x)),
      y: Math.max(radius + padding, Math.min(height - radius - padding, y))
    };
    for (let pass = 0; pass < 4; pass++) {
      svg.querySelectorAll('g.kg-clickable').forEach((other) => {
        if (other === group) return;
        const otherPosition = nodePosition(other.dataset.nodeId);
        if (!otherPosition) return;
        const minDistance = radius + (Number(other.dataset.radius) || 26) + padding;
        let dx = point.x - otherPosition.x;
        let dy = point.y - otherPosition.y;
        let distance = Math.hypot(dx, dy);
        if (distance >= minDistance) return;
        if (distance < 0.001) { dx = 1; dy = 0; distance = 1; }
        point.x = otherPosition.x + (dx / distance) * minDistance;
        point.y = otherPosition.y + (dy / distance) * minDistance;
        point.x = Math.max(radius + padding, Math.min(width - radius - padding, point.x));
        point.y = Math.max(radius + padding, Math.min(height - radius - padding, point.y));
      });
    }
    return point;
  };
  svg.addEventListener('pointerdown', (event) => {
    const group = event.target.closest('g.kg-clickable');
    if (!group) return;
    const ctm = svg.getScreenCTM();
    if (!ctm) return;
    const pt = new DOMPoint(event.clientX, event.clientY).matrixTransform(ctm.inverse());
    const position = nodePosition(group.getAttribute('data-node-id'));
    dragNodeId = group.getAttribute('data-node-id');
    dragOffset = { dx: pt.x - (position?.x || 0), dy: pt.y - (position?.y || 0) };
    if (group.setPointerCapture) group.setPointerCapture(event.pointerId);
  });
  svg.addEventListener('pointermove', (event) => {
    if (!dragNodeId) return;
    const ctm = svg.getScreenCTM();
    if (!ctm) return;
    const pt = new DOMPoint(event.clientX, event.clientY).matrixTransform(ctm.inverse());
    const group = svg.querySelector(`g.kg-clickable[data-node-id="${CSS.escape(dragNodeId)}"]`);
    if (!group) return;
    const point = constrainNodePosition(dragNodeId, pt.x - dragOffset.dx, pt.y - dragOffset.dy);
    group.setAttribute('transform', `translate(${point.x.toFixed(1)},${point.y.toFixed(1)})`);
    updateLinkedEdges(dragNodeId);
  });
  const endDrag = () => { dragNodeId = null; };
  svg.addEventListener('pointerup', endDrag);
  svg.addEventListener('pointercancel', endDrag);

  if (diffMode) scheduleKgDiffReveal(container);
}

const kgRevealObservers = new WeakMap();

// 差异视图：每次模块进入视口时，新增节点逐个出现，随后新增边逐条出现。
function scheduleKgDiffReveal(container) {
  const previous = kgRevealObservers.get(container);
  if (previous) { previous.disconnect(); kgRevealObservers.delete(container); }
  const newNodes = Array.from(container.querySelectorAll('g.kg-clickable.is-new'));
  const newLinks = Array.from(container.querySelectorAll('g.kg-link.is-new'));
  const sequence = [...newNodes, ...newLinks];
  if (!sequence.length) return;
  sequence.forEach((el) => el.classList.add('kg-pending'));
  const STEP_MS = 140;
  let timer = null;
  const reveal = () => {
    if (timer) clearTimeout(timer);
    sequence.forEach((el) => {
      el.classList.remove('kg-shown');
      el.classList.add('kg-pending');
    });
    let index = 0;
    const tick = () => {
      if (index >= sequence.length) return;
      const el = sequence[index];
      el.classList.remove('kg-pending');
      el.classList.add('kg-shown');
      index += 1;
      timer = setTimeout(tick, STEP_MS);
    };
    tick();
  };
  const observer = new IntersectionObserver((entries) => {
    if (entries.some((entry) => entry.isIntersecting)) reveal();
  }, { threshold: 0.08 });
  observer.observe(container);
  kgRevealObservers.set(container, observer);
}

const TEAM_NODE_RADIUS = {
  Principle: 40, Mechanism: 36, RootCause: 32, Symptom: 30, FaultGroup: 30,
  FaultMode: 28, BoilerEntity: 30, Constraint: 26, Regime: 26, ModelPrior: 26,
  SensorNoise: 24, TimeWindow: 22, Variable: 20
};
const TEAM_EDGE_FAMILIES = {
  BELONGS_TO: 'belongs', GOVERNS: 'governs', CONSTRAINS: 'constrains',
  MANIFESTS_AS: 'manifests', STORES: 'stores', PROVIDES: 'stores',
  CAUSES: 'causal', DRIVES: 'causal', TRIGGERS: 'causal', AFFECTS: 'causal',
  INTENSIFIES: 'causal', INJECTS: 'causal', RAISES_RISK: 'causal', HURTS: 'causal',
  FEEDS: 'causal', DERIVED_FROM: 'causal', RELEASES: 'causal', RELAXES: 'causal',
  BENEFITS: 'causal', RELATES: 'relates', REQUIRES: 'relates', SENSITIVE_TO: 'relates',
  SUITED_FOR: 'relates', APPLIES_TO: 'relates', RESISTANT_TO: 'relates',
  CORRELATES_WITH: 'correlates'
};

let teamGraphState = { variables: false, correlation: false, threshold: 0.95 };
let teamLoading = null;
let teamDetailNodes = new Map();

function teamEdgeFamily(type) {
  return TEAM_EDGE_FAMILIES[type] || 'relates';
}

function computeTeamLayout(nodeList, edgeList, width, height) {
  const positions = new Map();
  const count = Math.max(nodeList.length, 1);
  const goldenAngle = Math.PI * (3 - Math.sqrt(5));
  nodeList.forEach((node, index) => {
    const radius = Math.min(width, height) * 0.34 * Math.sqrt(index / Math.max(count - 1, 1));
    const angle = index * goldenAngle;
    positions.set(node.id, {
      x: width / 2 + Math.cos(angle) * radius,
      y: height / 2 + Math.sin(angle) * radius,
      vx: 0,
      vy: 0
    });
  });
  const area = width * height;
  const k = Math.sqrt(area / Math.max(count, 1)) * (count > 180 ? 1.05 : 0.95);
  const iterations = count > 180 ? 280 : 340;
  const initialTemp = k;
  const repelCutoff = k * 2;
  const pad = 50;
  for (let iter = 0; iter < iterations; iter++) {
    const temperature = Math.max(initialTemp * (1 - iter / iterations), 1);
    for (let i = 0; i < nodeList.length; i++) {
      for (let j = i + 1; j < nodeList.length; j++) {
        const a = positions.get(nodeList[i].id);
        const b = positions.get(nodeList[j].id);
        const dx = b.x - a.x;
        const dy = b.y - a.y;
        const d2 = dx * dx + dy * dy || 1;
        const d = Math.sqrt(d2);
        if (d > repelCutoff) continue;
        const fr = (k * k) / d;
        const fx = (dx / d) * fr;
        const fy = (dy / d) * fr;
        a.vx -= fx;
        a.vy -= fy;
        b.vx += fx;
        b.vy += fy;
      }
    }
    edgeList.forEach((edge) => {
      const a = positions.get(edge.source);
      const b = positions.get(edge.target);
      if (!a || !b) return;
      const dx = b.x - a.x;
      const dy = b.y - a.y;
      const d = Math.sqrt(dx * dx + dy * dy) || 1;
      const fa = (d * d) / k;
      const fx = (dx / d) * fa;
      const fy = (dy / d) * fa;
      a.vx += fx;
      a.vy += fy;
      b.vx -= fx;
      b.vy -= fy;
    });
    for (const node of nodeList) {
      const p = positions.get(node.id);
      p.vx += (width / 2 - p.x) * 0.01;
      p.vy += (height / 2 - p.y) * 0.01;
      if (p.x < pad) p.vx += (pad - p.x) * 0.06;
      if (p.x > width - pad) p.vx -= (p.x - (width - pad)) * 0.06;
      if (p.y < pad) p.vy += (pad - p.y) * 0.06;
      if (p.y > height - pad) p.vy -= (p.y - (height - pad)) * 0.06;
      const speed = Math.hypot(p.vx, p.vy) || 1;
      const step = Math.min(speed, temperature);
      p.x += (p.vx / speed) * step;
      p.y += (p.vy / speed) * step;
      p.x = Math.max(pad, Math.min(width - pad, p.x));
      p.y = Math.max(pad, Math.min(height - pad, p.y));
      p.vx = 0;
      p.vy = 0;
    }
  }
  return positions;
}

function hideTeamDetail() {
  const existing = $('.kg-detail-card', document.querySelector('.kg-team-wrap'));
  if (existing) existing.remove();
}

function showTeamDetail(nodeId) {
  hideTeamDetail();
  const node = teamDetailNodes.get(nodeId);
  if (!node) return;
  const container = document.querySelector('.kg-team-wrap');
  const propKeys = ['dcs_code', 'range', 'var_group', 'type', 'category', 'verified_note'];
  const propRows = propKeys
    .filter((key) => node.props && node.props[key] !== undefined && node.props[key] !== '')
    .map((key) => `<dt>${escapeHtml(key)}</dt><dd>${escapeHtml(String(node.props[key]))}</dd>`)
    .join('');
  const card = document.createElement('div');
  card.className = 'kg-detail-card';
  card.innerHTML = `<button class="kg-detail-close" type="button" aria-label="关闭详情">×</button>
    <span class="kg-detail-type">${escapeHtml(node.type_label || node.type)}</span>
    <h3>${escapeHtml(node.name)}</h3>
    <div class="kg-detail-id">${escapeHtml(node.id)}</div>
    ${node.description ? `<p>${escapeHtml(node.description)}</p>` : ''}
    ${propRows ? `<dl>${propRows}</dl>` : ''}`;
  card.querySelector('.kg-detail-close').addEventListener('click', hideTeamDetail);
  container.append(card);
}

function renderTeamGraph(graph) {
  const container = document.querySelector('.kg-team-wrap');
  const nodes = graph.nodes || [];
  const edges = graph.edges || [];
  const summary = graph.summary || {};
  $('#teamSummary').textContent = `队友图谱：${summary.node_count} 节点 · ${summary.edge_count} 关系边${summary.correlation_shown ? `（含相关性边 ${summary.correlation_shown} 条）` : ''} · 导出于 ${summary.exported_at || '—'}`;
  if (!nodes.length) {
    container.innerHTML = '<div class="feature-empty"><h3>暂无可展示的知识图谱</h3><p>请确认 knowledge_graph/team/kg_snapshot.json 存在。</p></div>';
    return;
  }
  const nodesById = new Map(nodes.map((node) => [node.id, node]));
  const seenEdges = new Set();
  const uniqueEdges = edges.filter((edge) => {
    if (!nodesById.has(edge.source) || !nodesById.has(edge.target)) return false;
    const key = `${edge.source}::${edge.target}::${edge.type}`;
    if (seenEdges.has(key)) return false;
    seenEdges.add(key);
    return true;
  });
  const width = 1180;
  const height = 860;
  const positions = computeTeamLayout(nodes, uniqueEdges, width, height);
  const neighborMap = new Map();
  uniqueEdges.forEach((edge) => {
    if (!neighborMap.has(edge.source)) neighborMap.set(edge.source, new Set());
    if (!neighborMap.has(edge.target)) neighborMap.set(edge.target, new Set());
    neighborMap.get(edge.source).add(edge.target);
    neighborMap.get(edge.target).add(edge.source);
  });
  const isCorrelation = (edge) => edge.type === 'CORRELATES_WITH';
  const edgeMarkup = uniqueEdges.map((edge) => {
    const a = positions.get(edge.source);
    const b = positions.get(edge.target);
    if (!a || !b) return '';
    const midX = (a.x + b.x) / 2;
    const midY = (a.y + b.y) / 2;
    const family = teamEdgeFamily(edge.type);
    const rValue = isCorrelation(edge) ? `，|r|=${Math.abs(Number(edge.props?.pearson_r) || 0).toFixed(3)}` : '';
    const edgeTitle = `${edge.label}（${edge.type}）${rValue}：${nodesById.get(edge.source)?.name || edge.source} → ${nodesById.get(edge.target)?.name || edge.target}${edge.description ? `，${edge.description}` : ''}`;
    const visibleLabel = isCorrelation(edge) ? '' : `<text x="${midX.toFixed(1)}" y="${midY.toFixed(1)}" text-anchor="middle" class="kg-edge-label">${escapeHtml(edge.label)}</text>`;
    return `<g class="kg-link" data-source="${escapeHtml(edge.source)}" data-target="${escapeHtml(edge.target)}" data-correlation="${isCorrelation(edge)}">
      <line x1="${a.x.toFixed(1)}" y1="${a.y.toFixed(1)}" x2="${b.x.toFixed(1)}" y2="${b.y.toFixed(1)}" class="kg-edge kg-edge-team kg-edge-team-${family}${isCorrelation(edge) ? ' kg-edge-correlation' : ''}"/>
      ${visibleLabel}
      <title>${escapeHtml(edgeTitle)}</title>
    </g>`;
  }).join('');
  const nodeMarkup = nodes.map((node) => {
    const point = positions.get(node.id);
    if (!point) return '';
    const radius = TEAM_NODE_RADIUS[node.type] || 24;
    return `<g class="kg-clickable" data-node-id="${escapeHtml(node.id)}" data-type="${escapeHtml(node.type)}" transform="translate(${point.x.toFixed(1)},${point.y.toFixed(1)})">
      <circle r="${radius}" class="kg-node kg-node-${escapeHtml(node.type.toLowerCase())} kg-node-circle"/>
      <text y="${radius > 28 ? -6 : -3}" text-anchor="middle" class="kg-node-title">${escapeHtml(node.type_label || node.type)}</text>
      <text y="${radius > 28 ? 13 : 10}" text-anchor="middle" class="kg-node-meta">${escapeHtml(fitSvgText(node.name, 12))}</text>
      <title>${escapeHtml(`${node.type_label || node.type}：${node.name}`)}${node.description ? `，${escapeHtml(node.description)}` : ''}</title>
    </g>`;
  }).join('');
  teamDetailNodes = nodesById;
  container.innerHTML = `<div class="literature-legend">
    <span><i class="kg-dot kg-node-principle"></i>原理</span>
    <span><i class="kg-dot kg-node-mechanism"></i>机理</span>
    <span><i class="kg-dot kg-node-symptom"></i>现象</span>
    <span><i class="kg-dot kg-node-rootcause"></i>根因</span>
    <span><i class="kg-dot kg-node-faultgroup"></i>故障组</span>
    <span><i class="kg-dot kg-node-faultmode"></i>故障模式</span>
    <span><i class="kg-dot kg-node-boilerentity"></i>锅炉实体</span>
    <span><i class="kg-dot kg-node-constraint"></i>约束</span>
    <span><i class="kg-dot kg-node-regime"></i>工况</span>
    <span><i class="kg-dot kg-node-timewindow"></i>时间窗口</span>
    <span><i class="kg-dot kg-node-sensornoise"></i>传感器扰动</span>
    <span><i class="kg-dot kg-node-modelprior"></i>模型先验</span>
    <span><i class="kg-dot kg-node-variable"></i>变量</span>
  </div>
  <div class="kg-link-legend">
    <span><i class="kg-link-dot kg-edge-team-causal"></i>因果/影响</span>
    <span><i class="kg-link-dot kg-edge-team-belongs"></i>归属</span>
    <span><i class="kg-link-dot kg-edge-team-governs"></i>支配</span>
    <span><i class="kg-link-dot kg-edge-team-constrains"></i>约束</span>
    <span><i class="kg-link-dot kg-edge-team-manifests"></i>表现</span>
    <span><i class="kg-link-dot kg-edge-team-relates"></i>关联</span>
    <span><i class="kg-link-dot kg-edge-team-stores"></i>存储/供应</span>
    <span><i class="kg-link-dot kg-edge-team-correlates"></i>相关性</span>
  </div>
  <div class="kg-svg-scroll">
    <svg class="kg-svg" viewBox="0 0 ${width} ${height}" preserveAspectRatio="xMidYMid meet" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="锅炉机理-变量知识图谱">
      ${edgeMarkup}${nodeMarkup}
    </svg>
  </div>`;
  const svg = $('svg', container);
  const edgeElements = Array.from(svg.querySelectorAll('g.kg-link'));
  const clearGraphHighlight = () => {
    svg.classList.remove('has-hover');
    svg.querySelectorAll('.is-hover, .is-linked').forEach((el) => el.classList.remove('is-hover', 'is-linked'));
    edgeElements.forEach((el) => el.classList.remove('is-highlight'));
  };
  svg.addEventListener('mouseover', (event) => {
    const group = event.target.closest('g.kg-clickable');
    if (!group) return;
    svg.classList.add('has-hover');
    const id = group.getAttribute('data-node-id');
    group.classList.add('is-hover');
    (neighborMap.get(id) || []).forEach((neighborId) => {
      const neighbor = svg.querySelector(`g.kg-clickable[data-node-id="${CSS.escape(neighborId)}"]`);
      if (neighbor) neighbor.classList.add('is-linked');
    });
    edgeElements.forEach((el) => {
      if (el.dataset.source === id || el.dataset.target === id) el.classList.add('is-highlight');
    });
  });
  svg.addEventListener('mouseout', (event) => {
    const group = event.target.closest('g.kg-clickable');
    if (group && group.contains(event.relatedTarget)) return;
    clearGraphHighlight();
  });
  svg.addEventListener('click', (event) => {
    const target = event.target.closest ? event.target.closest('g[data-node-id]') : null;
    if (!target) { hideTeamDetail(); return; }
    showTeamDetail(target.getAttribute('data-node-id'));
  });
  let dragNodeId = null;
  let dragOffset = { dx: 0, dy: 0 };
  svg.addEventListener('pointerdown', (event) => {
    const group = event.target.closest('g.kg-clickable');
    if (!group) return;
    const ctm = svg.getScreenCTM();
    if (!ctm) return;
    const pt = new DOMPoint(event.clientX, event.clientY).matrixTransform(ctm.inverse());
    const match = /translate\(([\d.]+),([\d.]+)\)/.exec(group.getAttribute('transform') || '');
    dragNodeId = group.getAttribute('data-node-id');
    dragOffset = { dx: pt.x - (match ? parseFloat(match[1]) : 0), dy: pt.y - (match ? parseFloat(match[2]) : 0) };
    if (group.setPointerCapture) group.setPointerCapture(event.pointerId);
  });
  svg.addEventListener('pointermove', (event) => {
    if (!dragNodeId) return;
    const ctm = svg.getScreenCTM();
    if (!ctm) return;
    const pt = new DOMPoint(event.clientX, event.clientY).matrixTransform(ctm.inverse());
    const group = svg.querySelector(`g.kg-clickable[data-node-id="${CSS.escape(dragNodeId)}"]`);
    if (group) group.setAttribute('transform', `translate(${(pt.x - dragOffset.dx).toFixed(1)},${(pt.y - dragOffset.dy).toFixed(1)})`);
  });
  const endDrag = () => { dragNodeId = null; };
  svg.addEventListener('pointerup', endDrag);
  svg.addEventListener('pointercancel', endDrag);
}

async function loadTeamGraph(force = false) {
  if (teamLoading && !force) return teamLoading;
  const container = document.querySelector('.kg-team-wrap');
  $('#teamSummary').textContent = '正在读取队友知识图谱……';
  if (container) container.innerHTML = '<div class="feature-empty"><h3>正在构建知识图谱</h3><p>读取队友 Neo4j 快照并布局节点。</p></div>';
  teamLoading = (async () => {
    try {
      const graph = await api.knowledgeGraph('team', {
        include_variables: teamGraphState.variables ? '1' : '0',
        include_correlation: teamGraphState.correlation ? '1' : '0',
        corr_threshold: String(teamGraphState.threshold)
      });
      renderTeamGraph(graph);
    } catch (error) {
      $('#teamSummary').textContent = '知识图谱暂时无法同步';
      if (container) container.innerHTML = `<div class="feature-empty"><h3>无法读取知识图谱</h3><p>${escapeHtml(error.message)}</p></div>`;
    } finally { teamLoading = null; }
  })();
  return teamLoading;
}

let knowledgeLoading = null;
async function loadKnowledgeGraph(force = false) {
  if (knowledgeLoading && !force) return knowledgeLoading;
  $('#knowledgeGraph').innerHTML = '<div class="feature-empty"><h3>正在构建知识图谱</h3><p>读取真实研究历史和科学裁决。</p></div>';
  knowledgeLoading = (async () => {
    try {
      const [evolution] = await Promise.all([
        api.knowledgeGraph('evolution'),
        loadTeamGraph(force)
      ]);
      renderKnowledgeGraph(evolution);
    } catch (error) {
      $('#knowledgeSummary').textContent = '知识图谱暂时无法同步';
      $('#knowledgeGraph').innerHTML = `<div class="feature-empty"><h3>无法读取科研记忆</h3><p>${escapeHtml(error.message)}</p></div>`;
    } finally { knowledgeLoading = null; }
  })();
  return knowledgeLoading;
}

const KG_BASELINE_KEY = 'boilermind.kg.baseline.v1';

function loadKgBaseline(runId) {
  try {
    const store = JSON.parse(localStorage.getItem(KG_BASELINE_KEY) || '{}');
    const ids = store[runId];
    return Array.isArray(ids) ? new Set(ids) : null;
  } catch {
    return null;
  }
}

function saveKgBaseline(runId, ids) {
  try {
    const store = JSON.parse(localStorage.getItem(KG_BASELINE_KEY) || '{}');
    store[runId] = [...ids];
    localStorage.setItem(KG_BASELINE_KEY, JSON.stringify(store));
  } catch {
    // 存储不可用时仅跳过新增高亮，不影响图谱渲染。
  }
}

function captureKgBaseline(runId) {
  api.knowledgeGraph('evolution')
    .then((graph) => {
      saveKgBaseline(runId, (graph.nodes || []).map((node) => node.id));
    })
    .catch(() => {
      // 基线不可用时回退到“已查看节点”集合判断新增。
    });
}

const kgGrowthInflight = new Set();

function collectRunExperimentIds(state, research) {
  const experimentIds = new Set();
  const hypothesisIds = new Set();
  const add = (value, target) => {
    const text = String(value ?? '').trim();
    if (text) target.add(text);
  };
  for (const batch of state?.batches || []) {
    for (const member of batch?.members || []) {
      if (member?.status === 'COMPLETED' || member?.outcome) {
        add(member.experiment_id, experimentIds);
        add(member.hypothesis_id, hypothesisIds);
        add(member.outcome?.experiment_result?.experiment_id, experimentIds);
        add(member.outcome?.experiment_result?.hypothesis_id, hypothesisIds);
        add(member.outcome?.scientific_result?.experiment_id, experimentIds);
        add(member.outcome?.scientific_result?.hypothesis_id, hypothesisIds);
      }
    }
  }
  // 兜底：frontend 投影中的汇总实验/假设标识。
  add(research.verification?.experiment_id, experimentIds);
  add(research.raw?.execution?.experiment_id, experimentIds);
  add(research.verification?.hypothesis_id, hypothesisIds);
  return { experimentIds, hypothesisIds };
}

function locateRunGraphIds(graph, { experimentIds, hypothesisIds }) {
  const selectedNodes = new Set();
  (graph.nodes || []).forEach((node) => {
    if (node.type === 'ExperimentResult' && experimentIds.has(node.id)) selectedNodes.add(node.id);
    if (node.type === 'Hypothesis' && hypothesisIds.has(node.id)) selectedNodes.add(node.id);
    if (node.type === 'ValidatedHypothesis'
      && (experimentIds.has(node.experiment_id) || (node.hypothesis_id && hypothesisIds.has(node.hypothesis_id)))) {
      selectedNodes.add(node.id);
    }
  });
  // 沿关系补齐：实验 → 已验证假设(generates)；假设 ← 实验(validated_by)。
  (graph.edges || []).forEach((edge) => {
    if (edge.type === 'validated_by' && experimentIds.has(edge.target) && !selectedNodes.has(edge.source)) {
      selectedNodes.add(edge.source);
    }
    if (edge.type === 'generates' && experimentIds.has(edge.source) && !selectedNodes.has(edge.target)) {
      selectedNodes.add(edge.target);
    }
  });
  const selectedEdges = new Set();
  (graph.edges || []).forEach((edge) => {
    const key = `${edge.source}::${edge.type}::${edge.target}`;
    if (selectedNodes.has(edge.source) && selectedNodes.has(edge.target)) selectedEdges.add(key);
  });
  return { selectedNodes, selectedEdges };
}

async function mountKnowledgeGrowth(element, research) {
  const runId = research.runId;
  if (!runId || !research.completed) return;
  const slot = element?.querySelector('[data-kg-growth]');
  if (!slot) return;
  const body = slot.querySelector('[data-kg-growth-body]');
  const summary = slot.querySelector('[data-kg-growth-summary]');
  if (!body || body.querySelector('svg.kg-svg')) return;
  if (kgGrowthInflight.has(runId)) return;
  kgGrowthInflight.add(runId);
  try {
    const graph = await api.knowledgeGraph('evolution');
    const totalNodes = graph.summary?.node_count ?? (graph.nodes || []).length;
    const totalEdges = graph.summary?.edge_count ?? (graph.edges || []).length;
    const state = await api.researchV2(runId).catch(() => null);
    const { experimentIds, hypothesisIds } = collectRunExperimentIds(state, research);
    let newNodeIds;
    let newEdgeKeys;
    let extraNodeIds;
    let summaryText;
    if (experimentIds.size || hypothesisIds.size) {
      const located = locateRunGraphIds(graph, { experimentIds, hypothesisIds });
      newNodeIds = located.selectedNodes;
      newEdgeKeys = located.selectedEdges;
      extraNodeIds = located.selectedNodes;
      summaryText = `本次实验新增 ${located.selectedNodes.size} 个节点 / ${located.selectedEdges.size} 条边 · 当前图谱共 ${totalNodes} 节点 / ${totalEdges} 边`;
    } else {
      const baseline = loadKgBaseline(runId) || loadSeenKgNodes();
      const currentIds = new Set((graph.nodes || []).map((node) => node.id));
      newNodeIds = baseline && baseline.size
        ? new Set([...currentIds].filter((id) => !baseline.has(id)))
        : new Set();
      newEdgeKeys = new Set();
      summaryText = `本次实验新增 ${newNodeIds.size} 个节点 · 当前图谱共 ${totalNodes} 节点 / ${totalEdges} 边`;
    }
    const renderGrowthGraph = (mode = 'diff') => {
      const options = { container: body, extraNodeIds, standalone: true };
      if (mode === 'diff') Object.assign(options, { newNodeIds, newEdgeKeys });
      renderKnowledgeGraph(graph, options);
      slot.dataset.kgGrowthMode = mode;
      slot.querySelectorAll('[data-kg-growth-mode]').forEach((button) => {
        const active = button.dataset.kgGrowthMode === mode;
        button.classList.toggle('is-active', active);
        button.setAttribute('aria-pressed', String(active));
      });
    };
    renderGrowthGraph();
    slot.querySelectorAll('[data-kg-growth-mode]').forEach((button) => {
      button.addEventListener('click', () => renderGrowthGraph(button.dataset.kgGrowthMode));
    });
    slot.querySelector('[data-kg-growth-replay]')?.addEventListener('click', () => renderGrowthGraph('diff'));
    if (summary) {
      summary.textContent = summaryText;
    }
  } catch (error) {
    console.error('[kg-growth]', error);
    if (summary) summary.textContent = '知识图谱暂时无法同步';
    if (body) body.innerHTML = `<div class="feature-empty"><h3>无法读取知识图谱</h3><p>${escapeHtml(error.message)}</p></div>`;
  } finally {
    kgGrowthInflight.delete(runId);
  }
}

function toast(message) {
  const item = document.createElement('div');
  item.className = 'toast';
  item.textContent = message;
  $('#toastRegion').append(item);
  setTimeout(() => item.remove(), 3500);
}

function resizeComposer() {
  elements.input.style.height = 'auto';
  elements.input.style.height = `${Math.min(elements.input.scrollHeight, 150)}px`;
  localStorage.setItem(CONFIG.storageKeys.draft, elements.input.value);
}

function setModeMenu(open) {
  elements.modeMenu.hidden = !open;
  elements.modeTrigger.setAttribute('aria-expanded', String(open));
}

function syncModeControl(value) {
  const label = value === 'research' ? '直接研究' : '对话';
  elements.mode.value = value;
  elements.modeTriggerLabel.textContent = label;
  elements.modeMenu.querySelectorAll('[data-mode-value]').forEach((button) => button.setAttribute('aria-checked', String(button.dataset.modeValue === value)));
}

function updateConnection(ready, text) {
  state.connectionReady = ready;
  const activeCount = [...state.activeRuns.values()].filter((run) => run.question?.trim() && !run.completed && !run.failed && (run.processRunning || run.rawStage === 'queued')).length + state.pendingRequests.size;
  elements.progressText.textContent = ready ? '后端已连接' : '后端未连接';
  elements.progressPill.dataset.view = 'chat';
  elements.drawerSystemText.textContent = text;
  elements.progressDot.className = `progress-dot ${activeCount ? 'live' : ready ? 'ready' : ''}`;
  elements.drawerSystemDot.className = `system-dot ${ready ? 'ready' : ''}`;
}

let backendProbeTimer = null;

async function probeBackend(forceCapabilities = false) {
  try {
    await api.health();
    if (forceCapabilities || !state.capabilities) {
      state.capabilities = await api.capabilities();
    }
    if (!state.connectionReady) {
      updateConnection(true, state.capabilities?.fullCloudResearchReady ? '研究能力已就绪' : '本地研究可用');
    }
    return true;
  } catch (error) {
    updateConnection(false, '后端离线');
    return false;
  }
}

function startBackendProbe() {
  if (backendProbeTimer) return;
  backendProbeTimer = setInterval(() => probeBackend(), 10000);
}

async function loadCapabilities() {
  await probeBackend(true);
  startBackendProbe();
}

function loadUnity(force = false) {
  if (!elements.unityStage || elements.unityStage.querySelector('iframe') && !force) return;
  const unityUrl = new URL(CONFIG.unityUrl, window.location.href);
  const activeRunId = state.selectedRunId || [...state.activeRuns.keys()].at(-1) || state.history?.[0]?.runId;
  if (activeRunId) unityUrl.searchParams.set('run_id', activeRunId);
  if (force) unityUrl.searchParams.set('reload', Date.now().toString());
  elements.unityStage.innerHTML = `<div class="unity-loading-copy">正在加载 Unity WebGL 运行环境……</div><iframe class="unity-frame" src="${unityUrl.toString()}" title="BoilerMind Unity 数字孪生" allow="fullscreen" loading="eager"></iframe>`;
  const unityFrame = elements.unityStage.querySelector('iframe');
  if (unityFrame) {
    unityFrame.addEventListener('load', () => {
      const loadingCopy = elements.unityStage.querySelector('.unity-loading-copy');
      if (loadingCopy) loadingCopy.remove();
    });
  }
}

async function loadHistory(query = '', status = '') {
  try {
    const result = await api.history({ query, status });
    state.history = result.items || [];
    renderHistoryList(elements.historyList, state.history.slice(0, 12));
    renderHistoryTable(elements.historyTable, state.history);
    elements.historyCount.textContent = String(result.total ?? state.history.length);
    if (!query && !status) resumeActiveRunsFromHistory(state.history);
  } catch (error) {
    elements.historyList.innerHTML = '<div class="empty-state">历史研究暂时不可用</div>';
    elements.historyTable.innerHTML = '<div class="empty-state">无法连接后端历史接口</div>';
  }
}

async function resumeActiveRunsFromHistory(items) {
  [...state.activeRuns.entries()].forEach(([runId, run]) => { if (!run.question?.trim()) state.activeRuns.delete(runId); });
  const activeItems = items.filter((item) => ['running', 'queued'].includes(item.status) && item.runId && item.question?.trim() && !state.activeRuns.has(item.runId)).slice(0, 4);
  await Promise.all(activeItems.map(async (item) => {
    try {
      const research = normalizeResearch(await api.research(item.runId), item.question);
      research.capabilities = state.capabilities;
      research.startedAt = item.startedAt ? new Date(item.startedAt).getTime() : Date.now();
      state.activeRuns.set(research.runId, research);
      if (research.processRunning || research.rawStage === 'queued') pollResearch(research.runId, null, research.question);
    } catch { /* 历史中的损坏任务不会阻断其他任务恢复 */ }
  }));
  updateProgressPill();
}

function conversationProcess(question, currentStage = 0, label = '正在拆解问题') {
  const titles = ['问题拆解', '证据与假设', '实验方案', '实验执行', '科学评价', '科研报告'];
  return {
    question,
    rawStage: label,
    currentStage,
    stages: titles.map((title, index) => ({ id: ['problem', 'evidence', 'plan', 'execution', 'evaluation', 'report'][index], title, status: index < currentStage ? 'completed' : index === currentStage ? 'active' : 'waiting' }))
  };
}

function selectProcess(research) {
  if (!research) return;
  state.selectedRunId = research.runId || null;
  elements.chatView?.classList?.add('has-process');
  $('#chatView').classList.add('has-process');
  renderProcessIndex(elements.processSidebar, research);
}

function hideWelcome() {
  elements.welcome.hidden = true;
}

function scrollIfNearBottom() {
  const gap = elements.scroll.scrollHeight - elements.scroll.scrollTop - elements.scroll.clientHeight;
  if (gap < 180) elements.scroll.scrollTop = elements.scroll.scrollHeight;
  else elements.newProgress.hidden = false;
}

async function sendQuestion(question, mode) {
  const backendReady = await probeBackend();
  if (!backendReady) {
    updateConnection(false, '后端离线');
    toast('后端未接通：请先启动 BoilerMind 后端（端口 8765），页面将自动重连');
    return;
  }
  hideWelcome();
  appendUserMessage(elements.stream, question, mode);
  state.messages.push({ role: 'user', content: question });
  scrollIfNearBottom();
  if (mode === 'research') { startResearch(question); return; }
  const loading = appendAssistantProgress(elements.stream, question);
  loading.dataset.question = question;
  const requestId = `ask_${Date.now().toString(36)}`;
  loading.dataset.requestId = requestId;
  state.pendingRequests.set(requestId, { id: requestId, question, element: loading, stage: '正在检索证据与组织回答' });
  $('#chatView').classList.add('has-process');
  renderProcessIndex(elements.processSidebar, conversationProcess(question, 1, '正在检索证据与生成假设'));
  updateProgressPill();
  scrollIfNearBottom();
  const progressStartedAt = Date.now();
  const progressTimer = setInterval(() => {
    const elapsed = loading.querySelector('[data-progress-elapsed]');
    if (elapsed) elapsed.textContent = `${Math.floor((Date.now() - progressStartedAt) / 1000)} 秒`;
  }, 1000);
  try {
    const response = await api.assistant(question, state.messages.slice(-8), state.attachments.map((item) => item.id).filter(Boolean), state.sessionId);
    clearInterval(progressTimer);
    renderAssistantMessage(loading, response);
    state.messages.push({ role: 'assistant', content: response.answer || '' });
    renderProcessIndex(elements.processSidebar, conversationProcess(question, 1, '证据问答已完成'));
    const researchQuestion = response.research_question_summary?.trim();
    if (researchQuestion) await startResearch(researchQuestion);
  } catch (error) {
    clearInterval(progressTimer);
    renderAssistantError(loading, error);
  } finally {
    state.pendingRequests.delete(requestId);
    updateProgressPill();
    scrollIfNearBottom();
  }
}

async function startResearch(question, extras = {}) {
  const loading = appendResearchLaunch(elements.stream, question);
  loading.dataset.question = question;
  $('#chatView').classList.add('has-process');
  renderProcessIndex(elements.processSidebar, conversationProcess(question, 0, '正在创建研究'));
  try {
    const created = await api.createResearch(question, state.sessionId, extras);
    const runId = created.run_id || created.runId;
    captureKgBaseline(runId);
    const raw = await api.research(runId);
    const research = normalizeResearch(raw, question);
    research.startedAt = Date.now();
    research.capabilities = state.capabilities;
    loading.remove();
    const element = createResearchElement(elements.stream, research);
    state.activeRuns.set(research.runId, research);
    selectProcess(research);
    rememberRun(research);
    pollResearch(research.runId, element, question);
    updateProgressPill();
  } catch (error) {
    renderAssistantError(loading, error);
  }
}

function pollResearch(runId, element, question) {
  if (state.pollers.has(runId)) return;
  const poll = async () => {
    try {
      const raw = await enrichCompletedRun(await api.research(runId));
      const research = normalizeResearch(raw, question);
      const previous = state.activeRuns.get(runId);
      research.startedAt = previous?.startedAt || Date.now();
      research.capabilities = state.capabilities;
      state.activeRuns.set(runId, research);
      elements.stream.querySelectorAll(`[data-run-id="${CSS.escape(runId)}"]`).forEach((runElement) => {
        captureIterationState(runElement, runId);
        updateResearchElement(runElement, research);
        mountKnowledgeGrowth(runElement, research);
        initIterationReplay(runElement, research);
      });
      if (state.selectedRunId === runId) renderProcessIndex(elements.processSidebar, research);
      updateProgressPill();
      scrollIfNearBottom();
      if (research.completed || research.failed || research.needsHumanReview) {
        clearInterval(state.pollers.get(runId));
        state.pollers.delete(runId);
        await loadHistory();
      }
    } catch (error) {
      updateConnection(false, '正在重新连接');
    }
  };
  poll();
  const timer = setInterval(poll, document.hidden ? CONFIG.backgroundPollIntervalMs : CONFIG.pollIntervalMs);
  state.pollers.set(runId, timer);
}

function updateProgressPill() {
  const active = [...state.activeRuns.values()].filter((run) => run.question?.trim() && !run.completed && !run.failed && (run.processRunning || run.rawStage === 'queued'));
  const count = active.length + state.pendingRequests.size;
  elements.progressText.textContent = state.connectionReady ? '后端已连接' : '后端未连接';
  elements.progressPill.dataset.view = 'chat';
  elements.progressDot.className = `progress-dot ${count ? 'live' : state.connectionReady ? 'ready' : ''}`;
}

async function enrichCompletedRun(raw) {
  if (raw.stage !== 'completed') return raw;
  try {
    const report = await api.report(raw.run_id || raw.runId);
    return {
      ...raw,
      ...report,
      final_report: report,
      verification_result: raw.verification_result || report.verification_result,
      hypothesis_evaluation: raw.hypothesis_evaluation || report.hypothesis_evaluation,
      deployment_gate_report: raw.deployment_gate_report || report.deployment_gate_report
    };
  } catch {
    return raw;
  }
}

async function openRun(runId, historyRow = null) {
  if (historyRow) {
    historyRow.classList.add('is-opening');
    await new Promise((resolve) => setTimeout(resolve, 240));
  }
  showView('chat');
  hideWelcome();
  const loading = appendLoadingMessage(elements.stream, '正在读取研究任务');
  loading.classList.add('history-run-loading');
  try {
    const raw = await enrichCompletedRun(await api.research(runId));
    const historyItem = state.history.find((item) => item.runId === runId);
    const research = normalizeResearch(raw, historyItem?.question || '');
    research.capabilities = state.capabilities;
    research.startedAt = historyItem?.startedAt ? new Date(historyItem.startedAt).getTime() : Date.now();
    loading.remove();
    elements.stream.querySelectorAll('.research-run, .research-launch').forEach((item) => item.remove());
    const element = createResearchElement(elements.stream, research);
    element.classList.add('history-run-reveal');
    mountKnowledgeGrowth(element, research);
    initIterationReplay(element, research);
    state.activeRuns.set(runId, research);
    selectProcess(research);
    if (research.processRunning || research.rawStage === 'queued') pollResearch(runId, element, research.question);
    elements.scroll.scrollTop = 0;
    elements.newProgress.hidden = true;
  } catch (error) { renderAssistantError(loading, error); }
}

async function openReport(runId) {
  showView('report');
  $('#viewTitle').textContent = '科研假设与研究计划';
  elements.reportReader.innerHTML = '<div class="empty-state">正在读取《科研假设与研究计划》……</div>';
  try {
    const report = await api.report(runId);
    state.selectedReport = report;
    const rationale = report?.results?.selection_interpretation || '';
    const rationaleCard = rationale ? `<div class="report-rationale-card"><span class="overline">模型选择</span><h3>模型选择缘由</h3><p>${escapeHtml(localizeReportText(rationale))}</p></div>` : '';
    const pdfUrl = `${CONFIG.apiBaseUrl}/api/v1/research-runs/${encodeURIComponent(runId)}/artifacts/scientific_plan_pdf/download?preview=1&v=20260829-reader-zh`;
    elements.reportReader.innerHTML = `${rationaleCard}<iframe class="report-pdf-frame" src="${pdfUrl}" title="科研假设与研究计划 PDF 预览"></iframe>`;
  } catch (error) {
    elements.reportReader.innerHTML = `<div class="empty-state">${error.message}</div>`;
  }
}

async function uploadFiles(files) {
  const pending = [...files].map((file) => ({ name: file.name, status: '上传中' }));
  state.attachments.push(...pending);
  renderAttachments();
  try {
    const result = await api.upload(files);
    const uploaded = result.attachments || result || [];
    pending.forEach((item, index) => Object.assign(item, { status: '已上传', id: uploaded[index]?.id || uploaded[index]?.attachmentId }));
  } catch (error) {
    pending.forEach((item) => { item.status = '上传失败'; });
    toast(error.message);
  }
  renderAttachments();
}

function renderAttachments() {
  elements.attachmentList.replaceChildren(...state.attachments.map((item) => {
    const chip = document.createElement('span');
    chip.className = 'attachment-chip';
    chip.textContent = `${item.name} · ${item.status}`;
    return chip;
  }));
}

document.addEventListener('click', async (event) => {
  const button = event.target.closest('button, [data-action]');
  if (!button) return;
  if (button.dataset.action === 'toggle-drawer') setDrawer(!elements.shell.classList.contains('drawer-open'));
  if (button.dataset.action === 'close-drawer') setDrawer(false);
  if (button.dataset.action === 'new-chat') { newSession(); elements.stream.innerHTML = ''; elements.welcome.hidden = false; elements.processSidebar.hidden = true; $('#chatView').classList.remove('has-process'); setDrawer(false); showView('chat'); }
  if (button.dataset.view) showView(button.dataset.view);
  if (button.dataset.runId && !button.dataset.action) await openRun(button.dataset.runId, button.closest('.history-row'));
  if (button.dataset.action === 'start-research') await startResearch(button.dataset.question || state.messages.at(-2)?.content || '');
  if (button.dataset.action === 'open-report') await openReport(button.dataset.runId);
  if (button.dataset.action === 'download-artifact' && button.dataset.url) window.open(`${CONFIG.apiBaseUrl}${button.dataset.url}`, '_blank', 'noopener');
  if (button.dataset.action === 'back-to-chat') showView('chat');
  if (button.dataset.action === 'return-current-conversation') { showView('chat'); elements.scroll.scrollTop = elements.scroll.scrollHeight; }
  if (button.dataset.action === 'return-pending') { showView('chat'); const pending = state.pendingRequests.get(button.dataset.requestId); pending?.element?.scrollIntoView({ block: 'center' }); }
  if (button.dataset.action === 'return-active-run') openRun(button.dataset.runId);
  if (button.dataset.action === 'focus-stage') {
    showView('chat');
    const runSelector = button.dataset.runId ? `[data-run-id="${CSS.escape(button.dataset.runId)}"]` : '.research-run:last-of-type';
    const runElement = elements.stream.querySelector(runSelector);
    const stageElement = runElement?.querySelector(`[data-stage="${CSS.escape(button.dataset.stageId)}"]`) || elements.stream.querySelector(`[data-stage="${CSS.escape(button.dataset.stageId)}"]`);
    if (stageElement) {
      const top = stageElement.getBoundingClientRect().top - elements.scroll.getBoundingClientRect().top + elements.scroll.scrollTop - 24;
      elements.scroll.scrollTo({ top, behavior: 'smooth' });
      stageElement.classList.add('stage-targeted');
      setTimeout(() => stageElement.classList.remove('stage-targeted'), 900);
    }
  }
  if (button.dataset.action === 'retry-assistant') sendQuestion(button.dataset.question, 'chat');
  if (button.dataset.action === 'refresh-history') loadHistory($('#fullHistorySearch').value, $('#historyStatus').value);
  if (button.dataset.action === 'refresh-knowledge') loadKnowledgeGraph(true);
  if (button.dataset.action === 'refresh-team-graph') loadTeamGraph(true);
  if (button.dataset.action === 'copy-raw-metrics') {
    const metrics = executionMetricsStore.get(button.dataset.runId);
    if (metrics !== undefined) {
      await navigator.clipboard.writeText(rawMetricsJson(metrics));
      toast('原始指标 JSON 已复制');
    } else {
      toast('暂无可复制的原始指标');
    }
  }
  if (button.dataset.kgTab) {
    document.querySelectorAll('.knowledge-tab').forEach((tab) => tab.classList.toggle('is-active', tab === button));
    document.querySelectorAll('[data-kg-panel]').forEach((panel) => {
      const active = panel.dataset.kgPanel === button.dataset.kgTab;
      panel.classList.toggle('is-active', active);
      panel.hidden = !active;
    });
  }
  if (button.dataset.action === 'copy-report') { await navigator.clipboard.writeText($('#reportConclusion')?.textContent || ''); toast('结论已复制'); }
  if (button.dataset.action === 'download-report' && state.selectedReport) { const blob = new Blob([JSON.stringify(state.selectedReport, null, 2)], { type: 'application/json' }); const link = document.createElement('a'); link.href = URL.createObjectURL(blob); link.download = `${state.selectedReport.run_id || 'boilermind-report'}.json`; link.click(); URL.revokeObjectURL(link.href); }
  if (button.dataset.action === 'download-plan-pdf') {
    const runId = state.selectedReport?.metadata?.run_id || state.selectedReport?.run_id;
    if (!runId) { toast('暂无可下载的 PDF 报告'); return; }
    window.open(`${CONFIG.apiBaseUrl}/api/v1/research-runs/${encodeURIComponent(runId)}/artifacts/scientific_plan_pdf/download`, '_blank', 'noopener');
  }
  if (button.dataset.action === 'select-hypothesis') await startResearch('', { resume_from: button.dataset.runId, select_hypothesis: button.dataset.hypothesisId });
  if (button.dataset.action === 'load-unity') loadUnity();
  if (button.dataset.action === 'reload-unity') loadUnity(true);
  if (button.id === 'modeTrigger') setModeMenu(elements.modeMenu.hidden);
  if (button.dataset.modeValue) { syncModeControl(button.dataset.modeValue); setMode(button.dataset.modeValue); setModeMenu(false); }
});

elements.composer.addEventListener('submit', async (event) => {
  event.preventDefault();
  const question = elements.input.value.trim();
  if (!question || elements.send.disabled) return;
  elements.input.value = '';
  resizeComposer();
  localStorage.removeItem(CONFIG.storageKeys.draft);
  sendQuestion(question, elements.mode.value);
});

elements.input.addEventListener('input', resizeComposer);
elements.input.addEventListener('keydown', (event) => { if (event.key === 'Enter' && !event.shiftKey) { event.preventDefault(); elements.composer.requestSubmit(); } });
elements.fileInput.addEventListener('change', () => { if (elements.fileInput.files.length) uploadFiles(elements.fileInput.files); elements.fileInput.value = ''; });
elements.newProgress.addEventListener('click', () => { elements.scroll.scrollTop = elements.scroll.scrollHeight; elements.newProgress.hidden = true; });
elements.scroll.addEventListener('scroll', () => { const gap = elements.scroll.scrollHeight - elements.scroll.scrollTop - elements.scroll.clientHeight; if (gap < 120) elements.newProgress.hidden = true; });
$('#historySearch').addEventListener('input', (event) => renderHistoryList(elements.historyList, state.history.filter((item) => (item.question || '').includes(event.target.value)).slice(0, 12)));
$('#fullHistorySearch').addEventListener('keydown', (event) => { if (event.key === 'Enter') loadHistory(event.target.value, $('#historyStatus').value); });
$('#historyStatus').addEventListener('change', (event) => loadHistory($('#fullHistorySearch').value, event.target.value));
$('#teamVarToggle').addEventListener('change', async (event) => {
  teamGraphState.variables = event.target.checked;
  const corrToggle = $('#teamCorrToggle');
  corrToggle.disabled = !teamGraphState.variables;
  if (!teamGraphState.variables) { teamGraphState.correlation = false; corrToggle.checked = false; }
  await loadTeamGraph(true);
});
$('#teamCorrToggle').addEventListener('change', async (event) => {
  teamGraphState.correlation = event.target.checked;
  await loadTeamGraph(true);
});
$('#teamCorrThreshold').addEventListener('change', async (event) => {
  teamGraphState.threshold = parseFloat(event.target.value);
  await loadTeamGraph(true);
});
$('#examplePrompts').addEventListener('click', (event) => {
  const prompt = event.target.closest('button');
  if (!prompt) return;
  elements.input.value = prompt.dataset.prompt || prompt.textContent.trim();
  setMode('research');
  syncModeControl('research');
  resizeComposer();
  elements.input.focus();
});
document.addEventListener('keydown', (event) => { if (event.key === 'Escape') { setDrawer(false); setModeMenu(false); } });
document.addEventListener('click', (event) => { if (!event.target.closest('#modeSelect')) setModeMenu(false); });

syncModeControl(state.mode);
elements.input.value = localStorage.getItem(CONFIG.storageKeys.draft) || '';
resizeComposer();
loadCapabilities();
loadHistory();

const initialView = location.hash.match(/^#\/(chat|unity|knowledge|history|report)$/)?.[1];
showView(initialView || 'chat');
window.addEventListener('hashchange', () => {
  const nextView = location.hash.match(/^#\/(chat|unity|knowledge|history|report)$/)?.[1];
  showView(nextView || 'chat');
});

const qaMode = new URLSearchParams(location.search).get('qa');
if (qaMode === 'active') {
  hideWelcome();
  const qaResearch = normalizeResearch({
    run_id: 'qa_active_visual_reference',
    stage: 'experiment_running',
    process_running: true,
    message: '设计验收状态，仅用于与选定视觉稿对照。',
    experiment_plan: { selected_solution: { name: 'DLinear 与 Persistence 对照实验' } },
    verification: { metrics: { sample_count: 25146, selected_model: 'DLinear' } }
  }, '在40%到60%深调工况下，能否提前10分钟可靠预测主蒸汽体积流量？');
  state.activeRuns.set(qaResearch.runId, qaResearch);
  createResearchElement(elements.stream, qaResearch);
  updateProgressPill();
} else if (qaMode === 'codex-chat') {
  hideWelcome();
  appendUserMessage(elements.stream, '打不开', 'chat');
  const qaReply = appendLoadingMessage(elements.stream);
  renderAssistantMessage(qaReply, {
    provider: '本地后端',
    answer: '### 结论\n\n**问题已经定位：你打开错端口了。**\n\n- `127.0.0.1:8765` 是后端服务，根路径没有网页。\n- 前端页面应打开：`http://127.0.0.1:8080/#/chat`\n\n### 检查结果\n\n1. 前端 8080 已连接正常。\n2. 后端 8765 已连接正常。',
    sources: [],
    hypothesis_ready: false
  });
} else if (qaMode === 'flow') {
  hideWelcome();
  const question = '在40%到60%深调工况下，能否提前10分钟可靠预测主蒸汽体积流量？';
  appendUserMessage(elements.stream, question, 'chat');
  const qaReply = appendLoadingMessage(elements.stream);
  renderAssistantMessage(qaReply, {
    provider: 'agent_bridge',
    answer: '### 证据结论\n\n已有文献支持进入完整实验，但最终可靠性必须通过真实历史数据、锁定测试集和 Persistence 基线验证。',
    sources: Array.from({ length: 13 }, (_, index) => ({ title: `锅炉预测证据 ${index + 1}`, url: `https://example.com/evidence-${index + 1}`, snippet: '用于验证证据抽屉的滚动与关闭交互。' })),
    hypothesis_ready: false,
    research_question_summary: question
  });
  appendResearchLaunch(elements.stream, question);
  $('#chatView').classList.add('has-process');
  renderProcessIndex(elements.processSidebar, conversationProcess(question, 1, '正在进入完整实验'));
} else if (qaMode === 'pending') {
  hideWelcome();
  const pending = appendAssistantProgress(elements.stream, '验证等待期间的任务导航是否清晰');
  pending.querySelector('[data-progress-elapsed]').textContent = '37 秒';
  const requestId = 'qa_pending_request';
  pending.dataset.requestId = requestId;
  state.pendingRequests.set(requestId, { id: requestId, question: '验证等待期间的任务导航是否清晰', element: pending, stage: '正在检索文献与组织回答' });
  $('#chatView').classList.add('has-process');
  renderProcessIndex(elements.processSidebar, conversationProcess('验证等待期间的任务导航是否清晰', 1, '正在检索证据'));
  updateProgressPill();
}
