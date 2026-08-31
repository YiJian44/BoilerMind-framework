# BoilerMind 前端交付包

这是一个纯静态前端，无需安装 Node.js、npm 或前端依赖。默认连接本机 BoilerMind 后端 `http://127.0.0.1:8765`。

## 一、首次运行

1. 安装 Python 3，安装时勾选 **Add Python to PATH**。
2. 完整解压本压缩包，不要在压缩包预览窗口中直接运行。
3. 在解压目录打开 PowerShell，执行：

   ```powershell
   powershell -ExecutionPolicy Bypass -File .\start-frontend.ps1
   ```

4. 浏览器会自动打开：`http://127.0.0.1:8080/#/chat`。

不能双击 `index.html` 通过 `file://` 打开；ES Module 和浏览器安全策略要求使用本地 HTTP 服务。

## 二、连接后端

### 前端和后端在同一台电脑

无需修改配置。先启动后端，并确认浏览器访问以下地址能够返回 JSON：

```text
http://127.0.0.1:8765/api/v1/capabilities
```

### 前端和后端在不同电脑

两台电脑需要位于可互通的局域网。假设后端电脑地址为 `192.168.1.20`：

```powershell
powershell -ExecutionPolicy Bypass -File .\configure-backend.ps1 -BackendUrl http://192.168.1.20:8765
```

配置后按 `Ctrl+F5` 刷新页面。后端还必须：

- 监听局域网地址，例如 `0.0.0.0:8765`，不能只监听 `127.0.0.1`；
- 在 Windows 防火墙中允许 8765 入站；
- 允许前端页面的 CORS 来源。

### 后端 CORS 配置

后端至少允许以下 Origin：

```text
http://127.0.0.1:8080
http://localhost:8080
```

至少允许 `GET, POST, OPTIONS` 和 `Content-Type` 请求头；如果启用了认证，还需允许实际使用的 `Authorization` 请求头。

## 三、联调验收

启动后依次检查：

1. 页面右上角显示“后端已连接”。
2. `GET /api/v1/capabilities` 返回 HTTP 200。
3. 对话模式能够调用 `POST /api/v1/assistant`。
4. 直接研究能够调用 `POST /api/v1/research-runs`。
5. 研究开始后能够轮询 `GET /api/v1/research-runs/{run_id}`。
6. 六阶段索引随真实状态推进，并显示当前任务、耗时、证据、实验指标和产物，而不是只显示最终结论。

完整后端契约见 [BACKEND_RESEARCH_PROGRESS_API_SPEC.md](./BACKEND_RESEARCH_PROGRESS_API_SPEC.md)。

## 四、常见问题

### PowerShell 提示禁止运行脚本

请使用以下命令。它只对本次启动放宽执行策略，不会永久修改系统设置：

```powershell
powershell -ExecutionPolicy Bypass -File .\start-frontend.ps1
```

### 页面显示“后端未连接”

- 确认后端进程和 8765 端口正常；
- 直接访问 `http://后端地址:8765/api/v1/capabilities`；
- 检查配置的后端 IP 和端口；
- 检查后端 CORS、监听地址和防火墙；
- 打开浏览器开发者工具，查看 Console 和 Network 中的首个错误。

### 8080 端口被占用

使用其他端口启动：

```powershell
powershell -ExecutionPolicy Bypass -File .\start-frontend.ps1 -Port 8081
```

同时在后端 CORS 中允许 `http://127.0.0.1:8081`。

### 问答长时间没有最终答案

当前问答后端是长请求，可能需要约 90～100 秒。等待期间页面会显示请求、证据检索和研究问题结构化过程。若最终超时，请检查后端日志和接口耗时。

### 如何停止前端服务

启动脚本会在后台运行 Python。可在任务管理器中结束对应 Python 进程；也可以在 PowerShell 中找到监听端口 8080 的进程后停止它。

## 五、可选 Unity WebGL

Unity 构建不包含在交付包内，也不影响研究对话和六阶段界面运行。如果已有 Unity WebGL 构建：

```powershell
.\start-frontend.ps1 -UnityRoot 'D:\path\to\UnityWebgl'
```

Unity 默认使用 8090 端口；实时状态同步仍要求对应的 Unity 桥接 WebSocket 服务。

## 六、项目结构

```text
index.html                          前端入口
assets/                             本地图标
css/                                页面样式
js/                                 前端逻辑和 API 配置
vendor/                             本地 Chart.js
serve_frontend.py                   无缓存静态服务
start-frontend.ps1                  一键启动脚本
configure-backend.ps1               后端地址配置脚本
BACKEND_RESEARCH_PROGRESS_API_SPEC.md  后端接口规范
```

## 七、数据边界

- 正式页面只请求真实后端 API，不自动回退 Mock。
- 默认 API 地址在 `js/config.js` 中；推荐使用配置脚本修改。
- 浏览器仅保存会话 ID、草稿、输入模式和最近任务，研究结果以服务端为准。
- `?qa=active` 仅用于视觉验收，不属于正式业务流程。
