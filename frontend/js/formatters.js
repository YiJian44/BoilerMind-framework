export function escapeHtml(value = '') {
  return String(value).replace(/[&<>'"]/g, (char) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', "'": '&#39;', '"': '&quot;' }[char]));
}

function inlineMarkdown(value = '') {
  const code = [];
  let output = escapeHtml(value).replace(/`([^`]+)`/g, (_, content) => {
    code.push(`<code>${content}</code>`);
    return `\u0000CODE${code.length - 1}\u0000`;
  });
  output = output
    .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
    .replace(/\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)/g, '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>');
  return output.replace(/\u0000CODE(\d+)\u0000/g, (_, index) => code[Number(index)]);
}

export function safeRichText(value = '') {
  const lines = String(value).replace(/\r\n?/g, '\n').split('\n');
  const blocks = [];
  let paragraph = [];
  let listType = '';
  let listItems = [];
  let fenced = false;
  let codeLines = [];

  const flushParagraph = () => {
    if (!paragraph.length) return;
    blocks.push(`<p>${paragraph.map(inlineMarkdown).join('<br>')}</p>`);
    paragraph = [];
  };
  const flushList = () => {
    if (!listItems.length) return;
    blocks.push(`<${listType}>${listItems.map((item) => `<li>${inlineMarkdown(item)}</li>`).join('')}</${listType}>`);
    listType = '';
    listItems = [];
  };

  lines.forEach((line) => {
    if (/^```/.test(line.trim())) {
      flushParagraph();
      flushList();
      if (fenced) { blocks.push(`<pre><code>${escapeHtml(codeLines.join('\n'))}</code></pre>`); codeLines = []; }
      fenced = !fenced;
      return;
    }
    if (fenced) { codeLines.push(line); return; }
    const heading = line.match(/^(#{1,4})\s+(.+)$/);
    const unordered = line.match(/^\s*[-*+]\s+(.+)$/);
    const ordered = line.match(/^\s*\d+[.)]\s+(.+)$/);
    if (heading) {
      flushParagraph(); flushList();
      const level = Math.min(Math.max(heading[1].length, 2), 4);
      blocks.push(`<h${level}>${inlineMarkdown(heading[2])}</h${level}>`);
    } else if (unordered || ordered) {
      flushParagraph();
      const nextType = unordered ? 'ul' : 'ol';
      if (listType && listType !== nextType) flushList();
      listType = nextType;
      listItems.push((unordered || ordered)[1]);
    } else if (!line.trim()) {
      flushParagraph(); flushList();
    } else {
      flushList();
      paragraph.push(line.trim());
    }
  });
  if (fenced && codeLines.length) blocks.push(`<pre><code>${escapeHtml(codeLines.join('\n'))}</code></pre>`);
  flushParagraph();
  flushList();
  return blocks.join('');
}

export function formatTime(value) {
  if (!value) return '—';
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return new Intl.DateTimeFormat('zh-CN', { month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit' }).format(date);
}

export function shortTitle(question = '', fallback = '未命名研究') {
  const compact = question.replace(/\s+/g, ' ').trim();
  return compact.length > 24 ? `${compact.slice(0, 24)}…` : compact || fallback;
}

import { researchRunStatus, scientificVerdict } from './i18n.js';

export function statusText(status) {
  return researchRunStatus(status) || status || '未知';
}

export function scientificText(value) {
  if (!value) return '尚未评价';
  const translated = scientificVerdict(value);
  if (translated === value) {
    return ({ refuted: '假设被反驳', inconclusive: '证据不足' })[value] || value;
  }
  return translated;
}

export function deploymentText(value) {
  return ({ allow: '允许部署', approved: '允许部署', block: '阻止部署', review: '需要人工审核', scientifically_valid: '科学有效' })[value] || value || '不适用';
}

export const METRIC_LABELS = {
  MAE: { label: 'MAE', unit: 't/h' },
  mae_t_h: { label: 'MAE', unit: 't/h' },
  RMSE: { label: 'RMSE', unit: 't/h' },
  rmse_t_h: { label: 'RMSE', unit: 't/h' },
  MBE: { label: 'MBE', unit: 't/h' },
  mbe_t_h: { label: 'MBE', unit: 't/h' },
  ACHIEVED_RISE_PCT: { label: '实际提升幅度', unit: '%' },
  PRESSURE_MAX_MPA: { label: '最大汽包压力', unit: 'MPa' },
  persistence_primary: { label: 'Persistence 基线（主指标）', unit: '' },
  selected_model_primary: { label: '选中模型（主指标）', unit: '' },
};

const METRIC_NOISE_KEYS = new Set([
  'model_rows', 'validation_metrics', 'locked_test_metrics', 'model_records',
  'dataset_sha256', 'artifact_paths', 'artifact_provenance', 'checkpoint_path',
  'checkpoint_compatible', 'generated_model', 'training_mode', 'failure_reason',
  'warnings', 'model_configuration', 'runtime_seconds', 'selected_model',
  'locked_test_best_model', 'primary_metric', 'metric_unit', 'python_version',
  'numpy_version', 'pandas_version', 'sklearn_version', 'torch_version',
  'random_seed', 'experiment_adapter', 'sample_count', 'train_samples',
  'validation_samples', 'test_samples', 'overall_sample_count',
  'total_elapsed_seconds', 'elapsed_seconds', 'started_at', 'completed_at',
  'R2', 'r2',
]);

function humanizeMetricKey(key) {
  return String(key)
    .replace(/_/g, ' ')
    .replace(/\b[a-z]/g, (char) => char.toUpperCase());
}

export function semanticMetricEntries(metrics) {
  const source = metrics || {};
  const preferredLabels = ['MAE', 'RMSE', 'MBE'];
  const grouped = new Map();
  for (const [key, value] of Object.entries(source)) {
    if (METRIC_NOISE_KEYS.has(key)) continue;
    if (value === undefined || value === null || typeof value === 'object') continue;
    const info = METRIC_LABELS[key] || { label: humanizeMetricKey(key), unit: '' };
    const dedupeKey = METRIC_LABELS[key] ? info.label : key;
    const uppercase = /^[A-Z]/.test(key);
    const existing = grouped.get(dedupeKey);
    if (!existing || (uppercase && !existing.uppercase)) {
      grouped.set(dedupeKey, { key, label: info.label, unit: info.unit, value, uppercase });
    }
  }
  const entries = [...grouped.values()];
  entries.sort((a, b) => {
    const ai = preferredLabels.indexOf(a.label);
    const bi = preferredLabels.indexOf(b.label);
    if (ai >= 0 && bi >= 0) return ai - bi;
    if (ai >= 0) return -1;
    if (bi >= 0) return 1;
    return a.label.localeCompare(b.label, 'zh-CN');
  });
  return entries;
}

export function formatMetricDisplay(entry, digits = 3) {
  const number = Number(entry.value);
  const text = Number.isFinite(number) ? number.toFixed(digits) : String(entry.value);
  if (!entry.unit) return text;
  return entry.unit === '%' ? `${text}%` : `${text} ${entry.unit}`;
}

export function rawMetricsJson(metrics) {
  try {
    return JSON.stringify(metrics, null, 2);
  } catch {
    return String(metrics);
  }
}
