const runtimeApiUrl = new URLSearchParams(window.location.search).get('api');

export const CONFIG = Object.freeze({
  apiBaseUrl: runtimeApiUrl || 'http://127.0.0.1:8765',
  unityUrl: 'http://127.0.0.1:8090/index_unity_only.html',
  pollIntervalMs: 2000,
  backgroundPollIntervalMs: 7000,
  requestTimeoutMs: 95000,
  historyPageSize: 20,
  storageKeys: {
    sessionId: 'boilermind.sessionId',
    draft: 'boilermind.draft',
    mode: 'boilermind.mode',
    recentRuns: 'boilermind.recentRuns'
  }
});
