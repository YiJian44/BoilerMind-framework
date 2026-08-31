/**
 * UnityBridge — 前端 JavaScript 桥接库
 * 
 * 功能：
 * 1. 通过 WebSocket 连接 Python 状态广播服务器，实时接收工况状态变更
 * 2. 通过 SendMessage 将状态码传递给 Unity WebGL 实例
 * 3. 监听 Unity 发回的状态确认回调
 * 4. 自动重连 + 状态恢复
 * 
 * 用法：
 *   const bridge = new UnityBridge('ws://localhost:8765/ws', 'unityInstance');
 *   bridge.connect();
 *   bridge.onStateChange((data) => console.log('状态变更:', data));
 * 
 * 可嵌入任意前端框架（Vue / React / 纯 HTML）
 */
class UnityBridge {
    /**
     * @param {string} serverUrl  WebSocket 地址，如 ws://localhost:8765/ws
     * @param {string} unityInstanceName  Unity WebGL 实例的 JS 变量名（通常为 'unityInstance' 或 'myGameInstance'）
     * @param {object} options  可选配置
     * @param {number} options.reconnectInterval  重连间隔(ms)，默认 3000
     * @param {number} options.maxReconnectAttempts  最大重连次数，默认 Infinity
     * @param {string} options.gameObjectName  Unity 中接收消息的 GameObject 名称，默认 'WaterWallBridge'
     */
    constructor(serverUrl, unityInstanceName = 'unityInstance', options = {}) {
        this.serverUrl = serverUrl;
        this.unityInstanceName = unityInstanceName;
        this.gameObjectName = options.gameObjectName || 'WaterWallBridge';
        this.reconnectInterval = options.reconnectInterval || 1000;
        this.maxReconnectInterval = options.maxReconnectInterval || 30000;
        this.maxReconnectAttempts = options.maxReconnectAttempts || Infinity;
        
        this.ws = null;
        this.connected = false;
        this.currentState = null;
        this._reconnectAttempts = 0;
        this._pendingMessages = [];  // Unity 未就绪时缓存的消息
        this._ackedFlowIds = new Set();  // 已应答的控制指令 flow_id
        this._unityReadyPollTimer = null;
        this._listeners = {
            stateChange: [],
            connected: [],
            disconnected: [],
            error: [],
            stateConfirm: [],
            controlInstruction: [],
            unityStatus: [],
            rawMessage: [],
            chartData: []
        };
        
        // 暴露全局引用，供 .jslib 回调
        if (typeof window !== 'undefined') {
            window.UnityBridge = this;
        }
    }

    // =========================================================================
    // 连接管理
    // =========================================================================

    /** 建立 WebSocket 连接 */
    connect() {
        if (this.ws && (this.ws.readyState === WebSocket.OPEN || this.ws.readyState === WebSocket.CONNECTING)) {
            console.warn('[UnityBridge] 已连接或正在连接中');
            return;
        }

        try {
            this.ws = new WebSocket(this.serverUrl);
        } catch (e) {
            this._emit('error', { message: `WebSocket 创建失败: ${e.message}` });
            return;
        }

        this.ws.onopen = () => {
            console.log(`[UnityBridge] 已连接: ${this.serverUrl}`);
            this.connected = true;
            this._reconnectAttempts = 0;
            this._emit('connected', {});
            this.notifyUnityStatus('connected', '已连接');
        };

        this.ws.onmessage = (event) => {
            this._handleMessage(event.data);
        };

        this.ws.onclose = (event) => {
            console.log(`[UnityBridge] 连接断开 (code=${event.code})`);
            this.connected = false;
            this._emit('disconnected', { code: event.code });
            this.notifyUnityStatus('disconnected', '连接断开');
            this._scheduleReconnect();
        };

        this.ws.onerror = (event) => {
            console.error('[UnityBridge] WebSocket 错误', event);
            this._emit('error', { message: 'WebSocket 错误' });
            this.notifyUnityStatus('error', '连接错误');
        };
    }

    /** 断开连接 */
    disconnect() {
        this.maxReconnectAttempts = 0; // 阻止重连
        if (this.ws) {
            this.ws.close();
            this.ws = null;
        }
    }

    /** 获取连接状态 */
    isConnected() {
        return this.connected && this.ws && this.ws.readyState === WebSocket.OPEN;
    }

    _scheduleReconnect() {
        if (this._reconnectAttempts >= this.maxReconnectAttempts) {
            console.warn('[UnityBridge] 达到最大重连次数，停止重连');
            return;
        }
        this._reconnectAttempts++;
        // Exponential backoff: 1s → 2s → 4s → 8s → 16s → max 30s
        var delay = Math.min(
            this.reconnectInterval * Math.pow(2, this._reconnectAttempts - 1),
            this.maxReconnectInterval
        );
        console.log(`[UnityBridge] ${delay}ms 后重连 (${this._reconnectAttempts}/${this.maxReconnectAttempts})`);
        this.notifyUnityStatus('reconnecting', `重连中(${this._reconnectAttempts})...`);
        var self = this;
        setTimeout(() => self.connect(), delay);
    }

    // =========================================================================
    // 消息处理
    // =========================================================================

    _handleMessage(rawData) {
        this._emit('rawMessage', rawData);

        try {
            const data = JSON.parse(rawData);

            // v1 唯一科研链路：Unity 只读消费同一 run_id 的已审计终态。
            if ((data.schema_version === 'boilermind.research_run.v1' || data.schema_version === 'boilermind.research_run.v2') &&
                (data.status === 'COMPLETED' || data.status === 'COMPLETED_WITH_REPORT_WARNING')) {
                const outcome = data.task_outcome || {};
                const summary = {
                    type: 'question',
                    run_id: data.run_id,
                    severity: outcome.generalization_status === 'GENERALIZATION_ADVANTAGE_CONFIRMED' ? 'info' : 'warning',
                    title: `科研实验完成：${outcome.selected_candidate}`,
                    detail: `最佳模型 ${outcome.selected_model || '—'}；选择依据 validation MAE；${outcome.generalization_status}`
                };
                const unityInstance = this._getUnityInstance();
                if (unityInstance) unityInstance.SendMessage(this.gameObjectName, 'ReceiveQuestion', JSON.stringify(summary));
                else { this._pendingMessages.push(JSON.stringify(summary)); this._startUnityReadyPoll(); }
                this._emit('researchRun', data);
            }

            // 状态变更消息（与 type 消息互斥，避免双发）
            if (data.state !== undefined && data.type === undefined) {
                this.currentState = data;
                this._emit('stateChange', data);
                this.sendToUnity(data);
            }

            // 连接确认消息
            if (data.type === 'connected') {
                console.log('[UnityBridge] 服务器确认连接');
                // 如果服务器返回了当前状态，直接应用
                if (data.currentState !== undefined) {
                    this.currentState = data.currentState;
                    this._emit('stateChange', data.currentState);
                    this.sendToUnity(data.currentState);
                }
            }

            // 图表数据消息
            if (data.type === 'chartData') {
                console.log(`[UnityBridge] 收到图表数据: ${data.chartType}, 实际值:${data.actualValues?.length || 0}点, 预测值:${data.predictedValues?.length || 0}点`);
                this._emit('chartData', data);
                const unityInstance = this._getUnityInstance();
                if (unityInstance) {
                    const json = JSON.stringify(data);
                    unityInstance.SendMessage(this.gameObjectName, 'ReceiveChartData', json);
                } else {
                    this._pendingMessages.push(JSON.stringify(data));
                    this._startUnityReadyPoll();
                }
            }

            // 仪表盘数据消息
            if (data.type === 'gaugeData') {
                console.log(`[UnityBridge] 收到仪表盘数据: airflow=${data.airflow}, temp=${data.temperature}, pressure=${data.pressure}`);
                const unityInstance = this._getUnityInstance();
                if (unityInstance) {
                    const json = JSON.stringify(data);
                    unityInstance.SendMessage(this.gameObjectName, 'ReceiveGaugeData', json);
                } else {
                    this._pendingMessages.push(JSON.stringify(data));
                    this._startUnityReadyPoll();
                }
            }

            // 故障指令消息
            if (data.type === 'fault') {
                console.log(`[UnityBridge] 收到故障指令: code=${data.code}, detail=${data.detail || ''}`);
                const unityInstance = this._getUnityInstance();
                if (unityInstance) {
                    const json = JSON.stringify(data);
                    unityInstance.SendMessage(this.gameObjectName, 'ReceiveFaultData', json);
                } else {
                    this._pendingMessages.push(JSON.stringify(data));
                    this._startUnityReadyPoll();
                }
            }

            // 热力图数据消息
            if (data.type === 'thermal') {
                console.log(`[UnityBridge] 收到热力图数据: probes=${data.probes?.length || 0}`);
                const unityInstance = this._getUnityInstance();
                if (unityInstance) {
                    const json = JSON.stringify(data);
                    unityInstance.SendMessage(this.gameObjectName, 'ReceiveThermalData', json);
                } else {
                    this._pendingMessages.push(JSON.stringify(data));
                    this._startUnityReadyPoll();
                }
            }

            // 管道热力图数据消息
            if (data.type === 'pipeThermal') {
                console.log(`[UnityBridge] 收到管道热力图数据: probes=${data.probes?.length || 0}`);
                const unityInstance = this._getUnityInstance();
                if (unityInstance) {
                    const json = JSON.stringify(data);
                    unityInstance.SendMessage(this.gameObjectName, 'ReceivePipeThermalData', json);
                } else {
                    this._pendingMessages.push(JSON.stringify(data));
                    this._startUnityReadyPoll();
                }
            }

            // 锅炉模拟结果消息
            if (data.type === 'simResult') {
                console.log(`[UnityBridge] 收到模拟结果: D=${data.steam_output} t/h, T=${data.wall_temp}°C, state=${data.state_code}(${data.state_name})`);
                const unityInstance = this._getUnityInstance();
                if (unityInstance) {
                    const json = JSON.stringify(data);
                    unityInstance.SendMessage(this.gameObjectName, 'ReceiveSimResult', json);
                } else {
                    this._pendingMessages.push(JSON.stringify(data));
                    this._startUnityReadyPoll();
                }
            }

            // 目标蒸汽量消息
            if (data.type === 'targetResult') {
                console.log(`[UnityBridge] 收到目标蒸汽量: D_target=${data.target_steam} t/h, T=${data.wall_temp}°C, state=${data.state_code}(${data.state_name})`);
                const unityInstance = this._getUnityInstance();
                if (unityInstance) {
                    const json = JSON.stringify(data);
                    unityInstance.SendMessage(this.gameObjectName, 'ReceiveTargetResult', json);
                    this._handleControlInstruction(data);
                } else {
                    this._pendingMessages.push(JSON.stringify(data));
                    this._startUnityReadyPoll();
                }
            }

            // Unity 推送状态消息（状态流转 sent→received→executed→returned）
            if (data.type === 'unityStatus') {
                console.log(`[UnityBridge] 收到 Unity 推送状态: ${data.status}`);
                this._emit('unityStatus', data);
            }

            // 问题推送消息
            if (data.type === 'question') {
                console.log(`[UnityBridge] 收到问题推送: [${data.severity}] ${data.title}`);
                const unityInstance = this._getUnityInstance();
                if (unityInstance) {
                    const json = JSON.stringify(data);
                    console.log(
                        "[UnityBridge] sending question to Unity:",
                        json
                    );
                    document.title = "QUESTION_RECEIVED";
                    try {
                        unityInstance.SendMessage(
                            this.gameObjectName,
                            'ReceiveQuestion',
                            json
                        );
                        console.log("[UnityBridge] ReceiveQuestion SendMessage success");
                        document.title = "SENDMESSAGE_SUCCESS";
                    }
                    catch(e){
                        console.error("[UnityBridge] ReceiveQuestion SendMessage failed:", e);
                        document.title = "SENDMESSAGE_FAILED";
                    }
                } else {
                    this._pendingMessages.push(JSON.stringify(data));
                    this._startUnityReadyPoll();
                }
            }
        } catch (e) {
            console.warn('[UnityBridge] 消息解析失败:', rawData, e);
        }
    }

    // =========================================================================
    // 发送到 Unity
    // =========================================================================

    /**
     * 将状态数据发送到 Unity WebGL
     * @param {object} stateData  状态数据对象
     */
    sendToUnity(stateData) {
        const unityInstance = this._getUnityInstance();
        const json = typeof stateData === 'string' ? stateData : JSON.stringify(stateData);

        if (!unityInstance) {
            console.warn('[UnityBridge] Unity 实例未就绪，消息已缓存');
            this._pendingMessages.push(json);
            this._startUnityReadyPoll();
            return false;
        }

        try {
            unityInstance.SendMessage(this.gameObjectName, 'ReceiveMessage', json);
            return true;
        } catch (e) {
            console.error('[UnityBridge] SendMessage 失败:', e);
            return false;
        }
    }

    // =========================================================================
    // 控制指令闭环（BoilerMind 控制优化 → Unity）
    // =========================================================================

    /** 处理 BoilerMind 控制指令：转发给 Unity C# 并向上行确认 received */
    _handleControlInstruction(data) {
        if (!data || data.flow !== 'control_instruction') return;
        const flowId = data.flow_id || data.run_id || '';
        this._emit('controlInstruction', data);
        if (flowId && !this._ackedFlowIds.has(flowId)) {
            this._ackedFlowIds.add(flowId);
            console.log(`[UnityBridge] 控制指令 ${flowId} 已转发给 Unity，回执 unity_ack`);
            this.sendUnityAck(flowId);
        }
    }

    /** 将任意 JSON 消息通过现有 WebSocket 发回后端 */
    sendToBackend(message) {
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(typeof message === 'string' ? message : JSON.stringify(message));
            return true;
        }
        console.warn('[UnityBridge] WebSocket 未连接，消息未发送：', message);
        return false;
    }

    /** 向上行确认「Unity 已接收控制指令」（sent → received） */
    sendUnityAck(flowId) {
        return this.sendToBackend({
            type: 'unity_ack',
            flow_id: flowId,
            detail: 'Unity 页面已接收控制指令',
            timestamp: new Date().toISOString()
        });
    }

    /** 向上行确认「Unity 已执行控制调整」（received → executed） */
    sendUnityExecuted(flowId, detail = 'Unity 已执行控制调整') {
        return this.sendToBackend({
            type: 'unity_executed',
            flow_id: flowId,
            detail: detail,
            timestamp: new Date().toISOString()
        });
    }

    /** 向上行回传实际蒸汽体积量（executed → returned，触发第二层裁决） */
    sendUnityResult(flowId, actualVolume, notes = 'Unity 实测回传') {
        const volume = Number(actualVolume);
        if (!Number.isFinite(volume)) {
            console.error('[UnityBridge] sendUnityResult 需要数值 actualVolume');
            return false;
        }
        return this.sendToBackend({
            type: 'unity_result',
            flow_id: flowId,
            actual_volume: volume,
            notes: notes,
            timestamp: new Date().toISOString()
        });
    }

    /** 重放所有缓存消息（Unity 就绪后自动调用） */
    replayPending() {
        const unityInstance = this._getUnityInstance();
        if (!unityInstance) return;

        while (this._pendingMessages.length > 0) {
            const json = this._pendingMessages.shift();
            try {
                // 根据 type 选择正确的 Unity 方法
                const parsed = JSON.parse(json);
                let method = 'ReceiveMessage';
                if (parsed.type === 'chartData') method = 'ReceiveChartData';
                else if (parsed.type === 'gaugeData') method = 'ReceiveGaugeData';
                else if (parsed.type === 'fault') method = 'ReceiveFaultData';
                else if (parsed.type === 'thermal') method = 'ReceiveThermalData';
                else if (parsed.type === 'pipeThermal') method = 'ReceivePipeThermalData';
                else if (parsed.type === 'simResult') method = 'ReceiveSimResult';
                else if (parsed.type === 'targetResult') method = 'ReceiveTargetResult';
                else if (parsed.type === 'question') method = 'ReceiveQuestion';

                unityInstance.SendMessage(this.gameObjectName, method, json);
                console.log(`[UnityBridge] 已重放缓存消息 [${method}]:`, json.substring(0, 80));
                if (parsed.type === 'targetResult') {
                    this._handleControlInstruction(parsed);
                }
            } catch (e) {
                console.error('[UnityBridge] 重放失败:', e);
            }
        }
    }

    /** @private 轮询检测 Unity 是否就绪，就绪后自动重放 */
    _startUnityReadyPoll() {
        if (this._unityReadyPollTimer) return;
        this._unityReadyPollTimer = setInterval(() => {
            if (this._getUnityInstance() && this._pendingMessages.length > 0) {
                console.log(`[UnityBridge] Unity 已就绪，重放 ${this._pendingMessages.length} 条缓存消息`);
                this.replayPending();
                clearInterval(this._unityReadyPollTimer);
                this._unityReadyPollTimer = null;
            }
        }, 500);
    }

    /**
     * 向 Unity 发送重置指令
     */
    sendReset() {
        const unityInstance = this._getUnityInstance();
        if (unityInstance) {
            unityInstance.SendMessage(this.gameObjectName, 'ReceiveReset', '');
        }
    }

    /**
     * 向后端发送锅炉模拟参数（Unity → JS → REST POST → 后端计算 → WebSocket 广播结果）
     * 由 .jslib 的 JS_SendSimRequest 调用。
     * @param {string} json  包含 coal_feed/air_flow/water_flow/drum_pressure/slag_degree 的 JSON 字符串
     */
    sendSimRequest(json) {
        // 从 WebSocket 地址推导 REST API 地址
        const httpUrl = this.serverUrl
            .replace('ws://', 'http://')
            .replace('wss://', 'https://')
            .replace('/ws', '/api/simulate');

        console.log(`[UnityBridge] 发送模拟请求到 ${httpUrl}`);

        fetch(httpUrl, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: json
        }).then(r => r.json()).then(data => {
            console.log('[UnityBridge] 模拟请求成功:', data);
        }).catch(e => {
            console.error('[UnityBridge] 模拟请求失败:', e);
        });
    }

    /**
     * 向后端发送「计算目标蒸汽量」请求（Unity → JS → REST POST → /api/calculate_target）
     * 由 .jslib 的 JS_SendCalculateTarget 调用。
     * 后端立即返回目标蒸汽量，同时通过 WebSocket 广播 type=targetResult → ReceiveTargetResult。
     * @param {string} json  包含 coal_feed/air_flow/water_flow/drum_pressure/slag_degree 的 JSON 字符串
     */
    sendCalculateTarget(json) {
        const httpUrl = this.serverUrl
            .replace('ws://', 'http://')
            .replace('wss://', 'https://')
            .replace('/ws', '/api/calculate_target');

        console.log(`[UnityBridge] 发送计算目标请求到 ${httpUrl}`);

        fetch(httpUrl, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: json
        }).then(r => r.json()).then(data => {
            console.log('[UnityBridge] 计算目标成功:', data);
        }).catch(e => {
            console.error('[UnityBridge] 计算目标请求失败:', e);
        });
    }

    // =========================================================================
    // 文件推送企业微信
    // =========================================================================

    /**
     * 将文件（截图/CSV）推送到后端，由后端转发到企业微信
     * @param {string} filename  文件名
     * @param {string} base64Data  base64 编码的文件内容（不含 data: 前缀）
     * @param {string} mimeType  MIME 类型
     */
    sendFileToWechat(filename, base64Data, mimeType) {
        const httpUrl = this.serverUrl
            .replace('ws://', 'http://')
            .replace('wss://', 'https://')
            .replace('/ws', '/api/notify/file');

        console.log(`[UnityBridge] 推送文件到企业微信: ${filename} (${mimeType})`);

        fetch(httpUrl, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                filename: filename,
                base64: base64Data,
                mimeType: mimeType
            })
        }).then(r => r.json()).then(data => {
            console.log('[UnityBridge] 文件推送成功:', data);
        }).catch(e => {
            console.error('[UnityBridge] 文件推送失败:', e);
        });
    }

    // =========================================================================
    // 连接状态通知 Unity
    // =========================================================================

    /**
     * 通知 Unity 端 WebSocket 连接状态变化
     * @param {string} status  connected | disconnected | reconnecting | error
     * @param {string} message  人类可读的描述
     */
    notifyUnityStatus(status, message) {
        var unityInstance = this._getUnityInstance();
        if (unityInstance) {
            var json = JSON.stringify({
                type: 'connectionStatus',
                status: status,
                message: message,
                timestamp: Date.now()
            });
            try {
                unityInstance.SendMessage(this.gameObjectName, 'ReceiveConnectionStatus', json);
            } catch (e) {
                console.warn('[UnityBridge] 通知 Unity 连接状态失败:', e);
            }
        }
    }

    // =========================================================================
    // 文件下载 & 全屏切换（供 .jslib 调用）
    // =========================================================================

    /**
     * 触发浏览器文件下载（截图 / CSV 共用）
     * @param {string} filename  文件名（含扩展名）
     * @param {string} base64Data  base64 编码的文件内容（不含 data: 前缀）
     * @param {string} mimeType  MIME 类型，如 "image/png" / "text/csv"
     */
    downloadFile(filename, base64Data, mimeType) {
        try {
            // base64 → binary string
            var byteChars = atob(base64Data);
            var byteNumbers = new Array(byteChars.length);
            for (var i = 0; i < byteChars.length; i++) {
                byteNumbers[i] = byteChars.charCodeAt(i);
            }
            var byteArray = new Uint8Array(byteNumbers);
            var blob = new Blob([byteArray], { type: mimeType });
            var url = URL.createObjectURL(blob);

            var a = document.createElement('a');
            a.href = url;
            a.download = filename;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
            console.log(`[UnityBridge] 文件已下载: ${filename} (${byteArray.length} bytes)`);
        } catch (e) {
            console.error('[UnityBridge] 文件下载失败:', e);
        }
    }

    /**
     * 切换浏览器全屏
     */
    toggleFullscreen() {
        if (!document.fullscreenElement) {
            document.documentElement.requestFullscreen().catch(function(e) {
                console.warn('[UnityBridge] 全屏请求失败:', e);
            });
        } else {
            document.exitFullscreen();
        }
    }

    _getUnityInstance() {
        if (typeof window === 'undefined') return null;
        // 尝试直接变量名
        if (window[this.unityInstanceName]) return window[this.unityInstanceName];
        // 尝试常见名称
        if (window.unityInstance) return window.unityInstance;
        if (window.myGameInstance) return window.myGameInstance;
        return null;
    }

    // =========================================================================
    // Unity → JS 回调（由 .jslib 调用）
    // =========================================================================

    /** @private 由 WaterWallBridge.jslib 调用 */
    _onStateConfirm(json) {
        try {
            const data = JSON.parse(json);
            this._emit('stateConfirm', data);
        } catch (e) {
            this._emit('stateConfirm', json);
        }
    }

    // =========================================================================
    // 事件系统
    // =========================================================================

    /** 注册状态变更回调 */
    onStateChange(callback) { this._on('stateChange', callback); return this; }
    /** 注册连接成功回调 */
    onConnected(callback) { this._on('connected', callback); return this; }
    /** 注册断开连接回调 */
    onDisconnected(callback) { this._on('disconnected', callback); return this; }
    /** 注册错误回调 */
    onError(callback) { this._on('error', callback); return this; }
    /** 注册 Unity 状态确认回调 */
    onStateConfirm(callback) { this._on('stateConfirm', callback); return this; }
    /** 注册 BoilerMind 控制指令回调 */
    onControlInstruction(callback) { this._on('controlInstruction', callback); return this; }
    /** 注册 Unity 推送状态流转回调 */
    onUnityStatus(callback) { this._on('unityStatus', callback); return this; }
    /** 注册原始消息回调 */
    onRawMessage(callback) { this._on('rawMessage', callback); return this; }
    /** 注册图表数据回调 */
    onChartData(callback) { this._on('chartData', callback); return this; }

    _on(event, callback) {
        if (this._listeners[event]) {
            this._listeners[event].push(callback);
        }
    }

    _emit(event, data) {
        if (this._listeners[event]) {
            this._listeners[event].forEach(cb => {
                try { cb(data); } catch (e) { console.error(`[UnityBridge] 回调错误:`, e); }
            });
        }
    }

    // =========================================================================
    // 工具方法
    // =========================================================================

    /** 获取当前状态 */
    getCurrentState() { return this.currentState; }

    /** 手动发送任意消息到 Unity */
    sendCustomMessage(methodName, value = '') {
        const unityInstance = this._getUnityInstance();
        if (unityInstance) {
            unityInstance.SendMessage(this.gameObjectName, methodName, value);
        }
    }
}

// 支持 ES Module 导入
if (typeof module !== 'undefined' && module.exports) {
    module.exports = UnityBridge;
}
