import { CONFIG } from './config.js';

class ApiError extends Error {
  constructor(message, status = 0, payload = null) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.payload = payload;
  }
}

function unwrap(payload) {
  if (payload && typeof payload === 'object' && 'success' in payload) {
    if (!payload.success) throw new ApiError(payload.error?.message || payload.error || '请求未成功', 0, payload);
    return payload.data;
  }
  return payload;
}

async function request(path, options = {}) {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), options.timeoutMs || CONFIG.requestTimeoutMs);
  try {
    const response = await fetch(`${CONFIG.apiBaseUrl}${path}`, {
      ...options,
      headers: options.body instanceof FormData ? options.headers : { 'Content-Type': 'application/json', ...options.headers },
      signal: controller.signal
    });
    const payload = await response.json().catch(() => null);
    if (!response.ok) {
      const message = payload?.error?.message || payload?.error || payload?.message || `请求失败（HTTP ${response.status}）`;
      throw new ApiError(message, response.status, payload);
    }
    return unwrap(payload);
  } catch (error) {
    if (error.name === 'AbortError') throw new ApiError('请求超时，请检查后端状态');
    if (error instanceof ApiError) throw error;
    throw new ApiError('无法连接 BoilerMind 后端');
  } finally {
    clearTimeout(timeout);
  }
}

export const api = {
  health: () => request('/health/ready', { timeoutMs: 5000 }),
  capabilities: () => request('/api/v1/capabilities', { timeoutMs: 5000 }),
  assistant: (question, history, attachmentIds, sessionId) => request('/api/v1/assistant', {
    method: 'POST',
    body: JSON.stringify({ question, history, attachmentIds, sessionId }),
    timeoutMs: 300000
  }),
  createResearch: (question, sessionId, extras = {}) => request('/api/v1/research-runs', {
    method: 'POST',
    body: JSON.stringify({ question, ...(extras.run_id ? { run_id: extras.run_id } : {}) })
  }),
  research: (runId) => request(`/api/v1/research-runs/${encodeURIComponent(runId)}/frontend`, { timeoutMs: 12000 }),
  researchV2: (runId) => request(`/api/v1/research-runs/${encodeURIComponent(runId)}`, { timeoutMs: 12000 }),
  report: (runId) => request(`/api/v1/research-runs/${encodeURIComponent(runId)}/report`, { timeoutMs: 12000 }),
  history: ({ query = '', status = '', page = 1 } = {}) => {
    const params = new URLSearchParams({ page: String(page), pageSize: String(CONFIG.historyPageSize), sort: '-updatedAt' });
    if (query) params.set('query', query);
    if (status) params.set('status', status);
    return request(`/api/v1/research-runs?${params}`).then((result) => ({
      ...result,
      items: (result.items || []).map((item) => ({
        ...item,
        runId: item.run_id,
        startedAt: item.started_at,
        updatedAt: item.completed_at || item.started_at
      }))
    }));
  },
  artifacts: (runId) => request(`/api/v1/research-runs/${encodeURIComponent(runId)}/artifacts`, { timeoutMs: 12000 }),
  knowledgeGraph: (kind, params = {}) => {
    const query = Object.entries(params)
      .filter(([, value]) => value !== undefined && value !== null && value !== '')
      .map(([key, value]) => `${encodeURIComponent(key)}=${encodeURIComponent(value)}`)
      .join('&');
    return request(`/api/v1/knowledge-graph/${encodeURIComponent(kind)}${query ? `?${query}` : ''}`, { timeoutMs: 15000 });
  },
  upload: async (files) => {
    const form = new FormData();
    [...files].forEach((file) => form.append('files', file));
    return request('/api/v1/uploads', { method: 'POST', body: form, timeoutMs: 60000 });
  }
};

export { ApiError };
