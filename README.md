# 🎩 历史人物/学者对话 Agent

输入历史人物或学者的名字，系统会先通过**联网搜索 + LLM 识别**该人物的身份与学者类别，再自动采集其**本人著作/作品、他人撰写的传记/评传、所处历史阶段的权威史料**，构建知识库后，你就能以该人物的身份进行深度对话，学习其思想与方法。支持**苏格拉底式教学模式**。

聚焦人物：弗洛伊德、尼采、康德、亚当·斯密、凯恩斯、托克维尔、爱因斯坦、亚里士多德、司马迁、孔子等。

## ✨ 功能特性

- **人物智能识别**：支持哲学家、心理学家、经济学家、社会学家、历史学家、科学家、作家等
- **多类别提示词**：LLM 判断人物所属类别，自动选用对应学者思维方式的提示词；一个人物可匹配多个类别
- **精简资料搜索路线**：先识别人物 → 再搜本人著作、他人传记、历史阶段权威史料 → 最后喂给对话模型
- **多源数据采集**：预置资料库 + 多语言维基 + DuckDuckGo 网页 +（可选）视频字幕
- **全角色扮演**：人物以其本人立场和思维方式发言，不做现代价值评判，便于沉浸式学习
- **RAG 检索增强**：回答基于真实著作/资料原文，双后端自动选择（sentence-transformers 优先，TF-IDF 回退）
- **构建进度可视化**：人物构建时显示进度条，并展示当前采用的资料/来源分布
- **苏格拉底式教学**：切换后人物不再直接给答案，而是通过追问引导你自己思考
- **流式对话**：实时逐字输出，等待模型思考时先显示“🧠 正在思考...”，避免误判为网络超时；对话历史持久化到 SQLite，重新加载人物时自动恢复历史

## 🚀 快速开始

> 需要 **Python 3.10+**（项目在 3.12 下开发验证）。

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

> RAG 检索为双后端：torch/sentence-transformers 可用时用语义检索（质量最高）；不可用时自动回退到 TF-IDF（无需 torch，开箱即用）。两者都不影响项目运行。

### 2. 配置 API Key

复制 `.env.example` 为 `.env`，填入你的 LLM API Key（支持任何 OpenAI 兼容接口）：

```bash
cp .env.example .env
```

```ini
LLM_API_KEY=your_api_key_here
LLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
LLM_MODEL=qwen3-turbo
```

也支持通义千问、智谱 GLM、DeepSeek 等，只需修改 `LLM_BASE_URL` 和 `LLM_MODEL`。

> 🔒 **安全提醒**：`.env` 已被 `.gitignore` 排除，请**不要**手动把它加入 Git 或上传到 GitHub，
> 其中包含你的 API Key。仓库中只提交 `.env.example`（占位符 Key）。

### 3. 启动

```bash
python app.py
```

浏览器打开 http://127.0.0.1:7860

### 4. 使用

1. 在输入框输入历史人物/学者名字（如「弗洛伊德」「亚当·斯密」「亚里士多德」）
2. 点击「构建人物档案」，系统会先识别身份与类别，再自动搜索本人著作、传记、历史背景并构建档案
3. 档案构建完成后，在下方对话框开始对话
4. 勾选「苏格拉底式教学模式」可切换到引导式学习

## 📁 项目结构

```
historical_agent/
├── app.py                          # Gradio 主入口（Web UI）
├── config.py                       # 配置管理
├── requirements.txt
│
├── core/                           # 核心逻辑
│   ├── llm_client.py               # LLM 客户端（OpenAI 兼容，流式 + JSON）
│   ├── person_identifier.py        # 人物身份/类别识别（联网 + LLM）
│   ├── search_engine.py            # DuckDuckGo + 多语言维基百科 + 深度资料检索
│   ├── video_transcript.py         # yt-dlp 视频字幕提取（可选）
│   ├── text_processor.py           # 文本清洗/切分/去重
│   ├── rag_engine.py               # RAG 双后端（FAISS / TF-IDF）
│   ├── data_collector.py           # 多源统一采集入口
│   ├── character_builder.py        # 人物档案构建（LLM 提取）
│   └── chat_engine.py              # 对话引擎（RAG + 历史 + LLM）
│
├── prompts/                        # 提示词
│   ├── identification_prompt.py    # 人物识别（输出身份、类别、检索词）
│   ├── category_prompts.py         # 各学者类别提示词库
│   ├── extraction_prompts.py       # 档案提取（输出结构化 JSON）
│   ├── character_prompt.py         # 角色扮演 System Prompt
│   └── socratic_prompt.py          # 苏格拉底式教学
│
├── models/                         # 数据模型
│   ├── character.py                # 人物档案
│   └── document.py                 # 文档段落（含来源标签）
│
├── db/                             # 数据库层
│   ├── database.py                 # SQLite 初始化
│   └── repository.py               # 人物/文档/对话 CRUD
│
├── assets/
│   ├── works/                      # 预置著作全文（按人物分目录）
│   │   ├── 弗洛伊德/  (梦的解析节选)
│   │   └── 亚当·斯密/ (国富论节选)
│   └── person_registry.json        # 人物别名/著作目录映射
│
└── data/                           # 运行时生成（gitignore）
    ├── historical_agent.db         # SQLite 数据库
    ├── faiss_index/                # 向量索引
    └── cache/                      # 爬取/字幕缓存
```

## 🔧 数据源说明

| 数据源 | 模块 | 说明 |
|--------|------|------|
| 人物识别 | `person_identifier.py` | 先联网搜索候选片段，再让 LLM 判断标准身份、类别、检索词 |
| 本人著作 | `search_engine.py` | 按 LLM 生成的 `work_queries` 搜索著作/作品 |
| 传记评传 | `search_engine.py` | 按 `biography_queries` 搜索他人写的传记/回忆录 |
| 历史史料 | `search_engine.py` | 按 `history_queries` 搜索时代背景/官方史料/权威档案 |
| 多语言维基 | `search_engine.py` | 中/英维基补充生平与背景 |
| 预置著作库 | `assets/works/` | 离线可用的精选著作全文，质量最高 |
| 视频字幕 | `video_transcript.py` | 可选：yt-dlp 提取 YouTube/Bilibili 字幕（较慢，默认关闭） |

### 添加新的预置著作

在 `assets/works/<人物名>/` 下放入 `.txt` 文件（UTF-8 编码），每个文件是一篇著作。文件名会作为著作标题。然后在 `assets/person_registry.json` 中确保该人物已注册（含别名），并为有预置著作的人物设置 `works_dir` 字段。

## ⚙️ 配置项

主要配置在 `config.py` 和 `.env`：

| 配置 | 默认值 | 说明 |
|------|--------|------|
| `LLM_API_KEY` | - | LLM API 密钥（必填） |
| `LLM_BASE_URL` | dashscope | OpenAI 兼容接口地址 |
| `LLM_MODEL` | - | 模型名 |
| `LLM_TEMPERATURE` | 0.7 | 对话温度 |
| 视频字幕采集 | 关闭 | 后台可选能力，前端默认不启用 |
| `RAG.top_k` | 5 | 每次检索返回段落数 |

## 🛡️ 立场说明

本应用采用**全角色扮演**模式：历史人物/学者以其本人真实的历史立场和思维方式发言，不附加现代价值评判。这是为了让用户沉浸式学习其思想与方法。所有观点均为历史人物真实立场，不代表应用开发者立场。应用设有安全底线：不煽动现实暴力、不提供武器制造等技术指导。

## 🐛 常见问题

**Q: 启动报 "未检测到 LLM_API_KEY"**
A: 未配置 `.env` 文件或其中 Key 为空。参考「快速开始」第 2 步。

**Q: 构建档案很慢**
A: 当前前端默认不启用视频字幕采集；后台仍保留该能力，但普通构建不会触发。慢的原因通常是网络搜索或 LLM 识别/提取耗时。

**Q: RAG 用的是哪个后端？**
A: 自动选择。控制台日志会显示 `[FAISS]` 或 `[TF-IDF]`。torch 可用时用语义检索，否则回退到 TF-IDF（仍可正常工作）。

**Q: DuckDuckGo 搜索失败**
A: DuckDuckGo 偶尔限流，属正常现象，不影响其他数据源采集。

---

## 🪪 许可证

[MIT](LICENSE) —— 本仓库仅用于学习交流，请遵守开源协议与当地法律法规。

## 🙏 致谢

- 各开源项目：Gradio、FAISS、sentence-transformers、yt-dlp 等
