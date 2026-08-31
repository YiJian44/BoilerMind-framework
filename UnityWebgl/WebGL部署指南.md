# Unity WebGL 部署使用指南

> 面向部署/运维人员：如何把「锅炉数字孪生系统」启动起来，供浏览器访问。
> 本文所有命令均不含具体机器绝对路径，使用相对路径或环境变量，可移植到任意机器。

---

## 一、系统组成

本系统由**两个服务** + **一套静态构建产物**组成：

| 组件 | 作用 | 默认端口 |
|------|------|----------|
| WebGL 文件服务器（`serve_webgl.py`） | 托管 WebGL 构建产物与网页，供浏览器访问 | `8080` |
| 状态服务器（`state_server.py`） | REST API + WebSocket 实时推送、故障模拟、企业微信推送 | `8770` |
| WebGL 构建产物（`Build/` 目录） | Unity 打包输出的场景、脚本、资源 | — |

两者独立启动，互不依赖，但前端页面需要同时访问两者才能完整工作：
- 浏览器访问 WebGL 文件服务器获取页面与 3D 场景
- 前端通过 WebSocket 连接状态服务器接收实时数据

---

## 二、部署前置条件

- **Python 3.8+**（含 fastapi / uvicorn / pydantic）
- **brotli 解压库**（用于在线解压 `.unityweb` 压缩数据）
- WebGL 构建已输出（构建目录含 `Build/` 子目录）
- 网络上已被其他机器访问时，请确认防火墙放行 `8080`、`8770` 端口

### 安装依赖

```bash
pip install -r requirements.txt
```

或手动安装关键依赖：

```bash
pip install fastapi uvicorn pydantic brotli
```

验证：

```bash
python -c "import fastapi, uvicorn, pydantic, brotli; print('deps OK')"
```

---

## 三、一键启动（推荐）

一份部署目录内通常包含 `start_all.py` 与 `一键启动.bat` 脚本，脚本会：

1. 自动探测可用的 Python 解释器（含 Anaconda、标准安装等），跳过系统 `WindowsApps` 桩文件；
2. 自动检查依赖是否齐全，缺失时提示安装命令；
3. 检查 8080 / 8770 端口是否被占用；
4. 依序启动 WebGL 文件服务器（8080）与状态服务器（8770）；
5. 自动打开浏览器访问三维页面与后端测试面板。

启动后控制台会打印全部访问入口：

```
================================================
  All services started!

  WebGL:    http://localhost:8080
  Test:     http://localhost:8080/test_panel.html
  WeChat:   http://localhost:8770/wechat-config
  WebSocket: ws://localhost:8770/ws
================================================
```

> 端口可在 `start_all.py` 顶部常量 `WEB_PORT` / `WS_PORT` 修改。

---

## 四、手动启动（可选，便于排查）

两个服务器脚本均位于 `server/` 子目录。

### 4.1 启动 WebGL 文件服务器

```bash
python server/serve_webgl.py --dir <指向 Build 所在目录> --port 8080
```

### 4.2 启动状态服务器

```bash
python server/state_server.py --port 8770
```

启动成功输出（uvicorn 日志）：

```
启动服务器: 0.0.0.0:8770
检查点目录: (未配置)
轮询间隔: 5s
INFO:     Started server process
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8770 (Press CTRL+C to quit)
```

### 4.3 端口说明

| 服务 | 端口 | 说明 |
|------|------|------|
| WebGL 页面 | `8080` | http 访问 |
| WebSocket | `8770` /ws | 实时数据推送 |
| REST API | `8770` | 后端接入（供预测模型/外部系统调用） |

若调整状态服务器端口，请同步修改前端桥接代码中的连接地址（WebGL 位于 `unity-bridge.js`，编辑器模式位于 `WebSocketClient` 脚本）。

---

## 五、浏览器访问

打开浏览器访问 **`http://部署机器:8080`**：

- **首次加载**：显示工业风格加载界面（深蓝渐变 + 荧光进度条 + 加载提示轮播），加载完成后淡出进入 3D 场景；
- **右上角状态指示**：绿色圆点=已连接，红色=未连接，黄色脉冲=正在重连；
- **完整功能**：三维场景、折线图、仪表盘、热力图、管道热力图、多故障报警面板、蒸汽模拟面板、问题推送弹窗、工具栏（截图/全屏/导出 CSV）等。

### 蒸汽量模拟面板

点击场景中"蒸汽量模拟"按钮打开面板，操作流程：

1. 调节5个工况参数（给煤量/送风量/给水流量/汽包压力/结渣程度）
2. 点"开始计算"→ 计算目标蒸汽量 → 左侧显示目标值和偏差
3. 调节参数逼近目标 → 500ms节流后重算 → 左侧实时更新当前蒸汽量和偏差
4. 偏差 < 2% → 弹出"已达到预期" → 可采纳方案/继续微调/放弃重置
5. 右侧"系统推荐参数"面板显示推荐/当前/偏差三列对比，下方"假设说明"区可滚动显示每条方案的文字说明

| 模式 | `useBackendCalculation` | 说明 |
|------|------------------------|------|
| 演示模式（默认） | `false` | 本地热力学公式计算，无需后端 |
| 后端模式 | `true` | POST `/api/calculate_target` + `/api/simulate` → WebSocket 回传 |

后端也可直接通过 WebSocket 广播 `type=targetResult` 消息主动推送目标蒸汽量，前端自动接收并设置目标。支持多组推荐参数（`rec_*` + `rec1_*` ~ `rec4_*`），每组附带独立的假设说明文字（`rec_notes` / `rec1_notes` 等），操作员通过"下一个"按钮循环切换查看不同方案。详见《后端通讯手册》§3.7。

### 问题推送弹窗

后端通过 `POST /api/question` 或 WebSocket 广播 `type=question` 消息，Unity 前端在屏幕中央弹出问题卡片，操作员点击「已知晓」关闭。

- **严重程度**：`info`（青色色条）/ `warning`（橙色）/ `critical`（红色）
- **测试方式**：测试面板（`test_panel.html`）卡片 08「问题推送测试」，支持自定义标题/内容 + 预设场景（超温确认/泄漏确认/检修提醒）
- 详见《后端通讯手册》§3.5

### WebSocket 客户端消息中继

状态服务器支持客户端→客户端消息中继：任何 WebSocket 客户端发送的含 `type` 字段的 JSON 消息，服务器自动广播给除发送者外的所有客户端（含 Unity）。后端模型可直接连接 WS 推送 `targetResult`（多组推荐参数+假设说明）、`question`、`chartData` 等消息给 Unity，无需经过 REST API。详见《后端通讯手册》§8。

---

## 六、企业微信推送配置

系统支持向「一/多个企业微信群机器人」推送告警、截图、CSV。

### 6.1 网页配置界面（推荐）

访问状态服务器上的配置页：

```
http://localhost:8770/wechat-config
```

功能：
- **添加/删除**多个机器人，每个可独立命名、填写 Webhook URL、启停；
- **保存配置**：写入 `wechat_config.json`（与服务器脚本同目录），**即时生效，无需重启**；
- **测试全部**：向所有已启用机器人发送一条测试消息，逐个显示成功/失败；
- **清空列表**：一键禁用所有推送。

也可从测试面板（`http://localhost:8080/test_panel.html`）顶部「企业微信配置 →」链接进入。

### 6.2 获取企业微信群机器人 URL

在企业微信群内：**群机器人 → 添加机器人** → 获得形如

```
https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=xxxx-xxxx-xxxx
```

的 Webhook URL，粘贴到配置页即可。

### 6.3 推送内容

| 触发 | 推送内容 |
|------|---------|
| 工况进入过温/泄漏/爆管（状态码 1/2/3） | Markdown 告警卡片 |
| 故障代码 ≥ 2 | Markdown 故障告警 |
| 用户一键截图 | 截图（自动压缩为 JPG，≤2MB），同时本地仍下载原始 PNG |
| 用户导出故障 CSV | CSV 内容（以代码块形式，超长自动截断） |

---

## 七、常见问题排查

### Q1 页面卡在 "加载中..." 不动

- `Build/` 内 `.unityweb` 文件未找到或无法解压
- 请务必使用本系统的 `serve_webgl.py` 启动（内含 Brotli 在线解压），**不要**使用 `python -m http.server`
- 确认依赖已安装：`import brotli` 能通过
- F12 → Console 查看是否有 `Failed to load resource: 404`

### Q2 右上角显示「未连接」（红点）

- 确认状态服务器（8770）与 WebGL 服务器（8080）均已启动
- 确认 8770 端口未被占用：`netstat -ano | findstr 8770`
- F12 → Console 查看 WebSocket 相关日志
- 浏览器需能同时访问两个端口（跨端口）。若部署在服务器，访问地址用服务器 IP。

### Q3 图表 Y 轴数字不显示

- 不要对 axis 的 label/name 直接设置 `textStyle.color`
- 应通过主题级属性控制：`theme.axis.textColor`
- 确认图表字体字段已指向中文字体

### Q4 三维场景 / 图表中文显示为方块

- 运行时创建的中文 Text（非 TMP）必须指定中文字体（如 SimHei）
- WebGL 默认的 LegacyRuntime 无 CJK 字形，务必在 Inspector 中为相关脚本指定中文字体

---

## 八、典型部署流程速查

```
1. 拷贝整个部署目录到目标机器（含 Build/、server 脚本、启动脚本、依赖清单）
2. 安装 Python 依赖（pip install -r requirements.txt）
3. 双击一键启动脚本
4. 浏览器访问 http://<机器IP>:8080
5. （可选）企业微信群添加机器人 → 打开配置页填写 Webhook
```