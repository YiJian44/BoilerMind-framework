import { CONFIG } from './config.js';

function safeParse(value, fallback) {
  if (value === null || value === undefined || value === '') return fallback;
  try { return JSON.parse(value); } catch { return fallback; }
}

function createSessionId() {
  return `bm_${Date.now().toString(36)}_${crypto.randomUUID().slice(0, 8)}`;
}

const storedSession = localStorage.getItem(CONFIG.storageKeys.sessionId);

export const state = {
  sessionId: storedSession || createSessionId(),
  messages: [],
  attachments: [],
  activeRuns: new Map(),
  pendingRequests: new Map(),
  selectedRunId: null,
  pollers: new Map(),
  history: [],
  capabilities: null,
  connectionReady: false,
  selectedReport: null,
  mode: localStorage.getItem(CONFIG.storageKeys.mode) || 'chat',
  recentRuns: safeParse(localStorage.getItem(CONFIG.storageKeys.recentRuns), [])
};

localStorage.setItem(CONFIG.storageKeys.sessionId, state.sessionId);

export function newSession() {
  state.sessionId = createSessionId();
  state.messages = [];
  state.attachments = [];
  localStorage.setItem(CONFIG.storageKeys.sessionId, state.sessionId);
  return state.sessionId;
}

export function setMode(mode) {
  state.mode = mode;
  localStorage.setItem(CONFIG.storageKeys.mode, mode);
}

export function rememberRun(run) {
  const compact = { runId: run.runId || run.run_id, question: run.question || '', updatedAt: new Date().toISOString() };
  state.recentRuns = [compact, ...state.recentRuns.filter((item) => item.runId !== compact.runId)].slice(0, 12);
  localStorage.setItem(CONFIG.storageKeys.recentRuns, JSON.stringify(state.recentRuns));
}
