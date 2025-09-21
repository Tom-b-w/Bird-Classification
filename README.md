# 鸟类音频识别 (Bird Sound Recogniser)

这是一个基于多模型（包括 Whisper / Wav2Vec2 / Bird-MAE 等）和音频处理工具的鸟类声学识别与合成项目。项目集成了 Web 前端（Flask 应用）、音频预处理与混合、单/多标签识别器、以及用于生成训练/合成数据的工具脚本。

## 主要特性
- Web 界面（Flask）：用户注册、登录、上传音频、单/多鸟识别、游戏/地图展示与合成音频生成功能。
- 单标签识别（基于 Whisper 或自定义模型）
- 多标签识别（基于 Wav2Vec2 的多标签分类器，支持同时识别多种鸟类）
- 音频混合与生成工具：根据文本关键词合成多种鸟类混合音频，用于演示或数据增强
- 简化版 Bird-MAE 演示脚本：基于特征提取与统计规则进行快速识别演示

## 目录结构（重要文件）
- `app.py` - 主 Flask 应用，包含前端路由、上传、预测（单/多标签）、合成等接口。
- `multi-bird.py` / `multi-bird_recognition.py` - 多鸟识别与 Bird-MAE 简化版实现与演示脚本。
- `process_audio.py` - 音频处理、关键词提取与混合（生成合成音频）的工具集合。
- `config.py` - 配置（例如邮件/SMTP 配置）。
- `static/` - 静态资源（图片、声音、生成混合音频等）。
- `audio_origin/` - 原始鸟类音频素材（按种类划分）。
- `models/`、`Training_MODEL_NAME/` - 可能存放训练/预训练模型的目录。

## 快速开始（Windows PowerShell）
下面是一个最小化的安装与运行说明，假设你已安装 Python 3.8+ 与 pip，并在项目根目录 `d:\Bird-Sound-Recogniser-main`。

1. 创建虚拟环境并激活（PowerShell）

    python -m venv .venv; .\.venv\Scripts\Activate.ps1

2. 安装依赖（requirements.txt 未包含在仓库中时需手动安装常用库）

    pip install flask flask_sqlalchemy flask_mail numpy torch torchvision torchaudio librosa transformers pydub soundfile scipy jieba captcha

（注意：若使用 GPU，则请安装对应版本的 torch/torchaudio；某些 transformers 模型需要 `trust_remote_code=True`）

3. 准备模型权重与数据
- 将单标签/多标签模型权重放置在仓库根或 `Training_MODEL_NAME/` 指定位置。例如：
  - `final_wav2vec2_model.pth` 用于多标签识别
  - `Training_MODEL_NAME/whisper/...` 用于单标签/whisper 相关模型加载
- 确保 `static/mixed_audio/`、`static/uploads/` 等目录存在（`app.py` 在启动时也会创建 `uploads`）。

4. 运行 Flask 应用（开发环境）

    python .\Bird-Sound-Recogniser\app.py

默认监听 0.0.0.0:5000，可在浏览器访问 http://127.0.0.1:5000

## 常见使用场景
- 上传音频并选择“单鸟”识别：将音频文件发送到 `/predict`，选择 bird_count=1。
- 多鸟识别：选择 bird_count > 1 或 `more`，后端会调用基于 Wav2Vec2 的多标签模型并返回多个候选。
- 合成音频：在 Web 界面或调用 `/synthesize` 提交文本，项目会从 `audio_origin/birds` 中挑选对应种类的音频并混合，生成到 `static/mixed_audio/`，返回可以通过静态 URL 访问的音频链接。

## 注意事项与提示
- 邮件相关配置（在 `app.py` / `config.py`）：仓库中示例包含 QQ SMTP 配置与示例密码占位符，请务必使用你自己的 SMTP 凭据并避免把真实密码写进代码或公开仓库。
- 大模型（transformers）会在首次加载时下载权重，请保证网络可用并预留足够磁盘空间。
- 识别精度与阈值：多标签识别函数 `recognize_multiple_birds` 中默认阈值为 0.3，可根据数据与需求调整。
- 部分脚本（例如 Bird-MAE 简化版）为演示/调试用途，结果为启发式或占位实现，不代表最终生产级精度。建议用真实标注数据进行微调/训练。

## 开发建议与后续改进
- 添加 `requirements.txt` 或 `environment.yml` 来锁定依赖。
- 把敏感配置（例如邮件密码、API key）移入环境变量或 `.env` 并在 `config.py` 中读取。
- 提供单元测试与简单的 CI 流水线来保证关键接口（预测/合成）不会在提交时回归。
- 将模型加载与推理逻辑抽象成服务（例如 FastAPI + Gunicorn/uvicorn）以便生产部署与水平扩展。

## 致谢
项目整合了开源库：librosa、transformers、pydub、torch 等。感谢所有贡献的工具与模型作者。

---

如果你希望我把 README 翻译成英文版、生成 `requirements.txt`、或把 README 内容写进仓库根目录的其他格式（例如 `README_.rst`），告诉我下一步即可。 
