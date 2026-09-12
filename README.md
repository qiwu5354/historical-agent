# 🎩 历史人物/学者对话 Agent

输入历史人物或学者的名字，系统会先通过**联网搜索 + LLM 识别**该人物的身份与学者类别，再自动采集其**本人著作/作品、他人撰写的传记/评传、所处历史阶段的权威史料**，构建知识库后，你就能以该人物的身份进行深度对话，学习其思想与方法。支持**苏格拉底式教学模式**。

聚焦人物：弗洛伊德、尼采、康德、亚当·斯密、凯恩斯、托克维尔、爱因斯坦、亚里士多德、司马迁、孔子等。

## ✨ 功能特性

- **人物智能识别**：支持哲学家、心理学家、经济学家、社会学家、历史学家、科学家、作家等
- **多类别提示词**：LLM 判断人物所属类别，自动选用对应学者思维方式的提示词；一个人物可匹配多个类别
- **精简资料搜索路线**：先识别人物 → 再搜本人著作、他人传记、历史阶段权威史料 → 最后喂给对话模型
- **同名消歧**：输入的名字若指向多个同名人物，自动列出候选，确认后再构建，避免张冠李戴
- **深度网页抓取**：搜索结果不只是摘要，`web_fetcher` 会下载正文，让 RAG 基于真实文章内容检索
- **多源数据采集**：预置资料库 + 多语言维基 + DuckDuckGo 网页 +（可选）视频字幕
- **全角色扮演**：人物以其本人立场和思维方式发言，不做现代价值评判，便于沉浸式学习
- **可调学者维度**：领域模板（自然科学/社会科学/人文/工程/商学）、职业阶段（早期/中期/资深）、沟通风格（学术/教学/面向公众）可叠加定制
- **对话模式切换**：`academic`（严谨学术论证）/ `friend`（轻松对话）两种模式，可与苏格拉底式教学叠加
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

> ⚠️ **务必保留这两个开关**（`.env.example` 中已带默认值），否则很容易踩到「模型返回空内容」的坑：
>
> ```ini
> LLM_DISABLE_THINKING=1   # 关闭「思考模式」（默认开）
> LLM_MAX_TOKENS=4096      # 输出预算，长档案提取建议 ≥4096
> ```
>
> `qwen3.5+`/`qwen3.7-flash`/`qwen3.8`、`GLM-5` 等**混合思考模型默认开启思考**，
> 思考内容（`reasoning_content`）与正文（`content`）**共享 `max_tokens` 预算**。
> 预算被思考吃光时会出现 HTTP 200 但正文为空/被截断，日志报
> `Expecting value: line 1 column 1 (char 0)`。详见「常见问题」。

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
3. 若名字指向多个同名人物（如多个「王阳明」），会弹出候选列表，选择具体人物后继续
4. 档案构建完成后，在下方对话框开始对话
5. 勾选「苏格拉底式教学模式」可切换到引导式学习；「对话模式」可在学术论证 / 轻松对话间切换

## 📁 项目结构

```
historical_agent/
├── app.py                          # Gradio 主入口（Web UI）
├── launcher.py                     # 一键启动辅助脚本（检查 .env / 依赖 / 端口后启动）
├── 一键启动.bat                     # Windows 双击启动
├── config.py                       # 配置管理
├── requirements.txt
│
├── tests/                          # 回归测试
│   └── test_llm_client.py          # LLM 客户端：思考模式 / 空正文重试 / 容错解析
│
├── core/                           # 核心逻辑
│   ├── llm_client.py               # LLM 客户端（OpenAI 兼容，流式 + JSON）
│   ├── person_identifier.py        # 人物身份/类别识别（联网 + LLM，含同名消歧）
│   ├── search_engine.py            # DuckDuckGo + 多语言维基百科 + 深度资料检索
│   ├── web_fetcher.py              # 网页正文抓取（httpx + BeautifulSoup，深度 RAG 数据源）
│   ├── video_transcript.py         # yt-dlp 视频字幕提取（可选）
│   ├── text_processor.py           # 文本清洗/切分/去重
│   ├── rag_engine.py               # RAG 双后端（FAISS / TF-IDF）
│   ├── data_collector.py           # 多源统一采集入口
│   ├── character_builder.py        # 人物档案构建（LLM 提取）
│   └── chat_engine.py              # 对话引擎（RAG + 历史 + LLM + 对话模式）
│
├── prompts/                        # 提示词
│   ├── identification_prompt.py    # 人物识别（输出身份、类别、检索词、同名候选）
│   ├── category_prompts.py         # 各学者类别提示词库
│   ├── extraction_prompts.py       # 档案提取（输出结构化 JSON）
│   ├── character_prompt.py         # 角色扮演 System Prompt
│   ├── socratic_prompt.py          # 苏格拉底式教学
│   ├── domain_templates.py         # 五大领域提示词模板（术语/方法论/认识论/引用/语气）
│   ├── scholar_profile.py          # 学者档案维度（职业阶段/研究焦点/方法偏好/沟通风格）
│   ├── scholar_enricher.py         # 学者提示词增强器（融合领域模板 + 阶段 + 风格）
│   ├── career_stages.py            # 职业阶段修饰词（早期/中期/资深）
│   ├── communication_styles.py     # 沟通风格修饰词（学术/教学/公众）
│   ├── dialogue_modes.py           # 对话模式（academic / friend）
│   └── validation.py               # 提示词验证（静态校验 + 输出一致性自检清单）
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
| 网页正文 | `web_fetcher.py` | 把搜索到的 URL 下载并提取正文（非仅摘要），供深度 RAG 使用 |
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
| `LLM_TEMPERATURE` | 0.7 | 对话温度（档案提取等严肃任务内部固定 0.2） |
| `LLM_DISABLE_THINKING` | 1 | 关闭「思考模式」。混合思考模型（qwen3.5+/qwen3.7-flash、GLM-5 等）思考默认开启，会与正文争抢 `max_tokens`，故默认关闭；设为 `0` 恢复，但需同时把 `LLM_MAX_TOKENS` 调到 8192 以上 |
| `LLM_MAX_TOKENS` | 4096 | 单次回复 token 预算（思考 token 也计入）。档案提取要输出 `core_thought`(500字) 等长字段，不建议低于 4096 |
| `RAG.top_k` | 5 | 每次检索返回段落数 |
| 视频字幕采集 | 关闭 | 后台可选能力，前端默认不启用 |
| `HTTPS_PROXY` / `HTTP_PROXY` | 空 | 采集维基/DDG/网页所需的网络代理；**不配置则直连**，国内环境通常需要设置（如 `http://127.0.0.1:7890`） |

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

**Q: 日志报 `模型输出无法解析为 JSON: Expecting value: line 1 column 1 (char 0)`，且 `原始内容(前500字):` 后面一片空白**
A: 这不是网络问题，而是**模型返回了空正文**（HTTP 200，但 `message.content` 是空字符串，
所以 `json.loads("")` 报 line 1 column 1）。根因是**思考模式吃光了 `max_tokens` 预算**：

- `qwen3.5+` / `qwen3.7-flash` / `qwen3.8` / `GLM-5` 等是**混合思考模型，思考默认开启**；
- 思考内容走 `reasoning_content` 字段，但**与正文共享 `max_tokens` 预算**；
- 档案提取要输出 `core_thought`(500字) 等多个长字段，2048 的预算被思考吃光后，正文就没了。

修复方式（本仓库已内置）：

1. `.env` 里保持 `LLM_DISABLE_THINKING=1`：客户端会自动关闭思考模式
   （DashScope 发 `enable_thinking=false`，DeepSeek 官方端点发 `thinking={"type":"disabled"}`，
   其它第三方端点自动跳过该参数，不会因参数不识别报 400）；
2. `LLM_MAX_TOKENS` 保持 `4096` 或更高；
3. 即使遇到空正文，客户端也会**自动加大 `max_tokens` 并追加 `/no_think` 重试一次**，
   仍失败则抛出带 `finish_reason` / 思考长度 / token 用量的明确错误，便于定位；
4. 若用的是**纯思考模型**（`qwen3-*-thinking`、`deepseek-r1` 等，思考不可关闭），
   请换成混合思考模型，或把 `LLM_MAX_TOKENS` 提到 8192 以上。

**Q: 日志显示 `共 0 段资料`、来源分布全是 0**
A: 这是采集环节没抓到任何资料（维基 / DuckDuckGo / 网页抓取全线为 0），随后会退化为
「用 LLM 内置知识构建（质量较低）」。通常是网络无法直连维基/DDG，请配置代理
（`HTTPS_PROXY=http://127.0.0.1:7890`，端口按你自己的代理软件填），并确认代理进程已启动。

## ✅ 测试

```bash
python -m pytest tests -q
```

覆盖 LLM 客户端最易出错的几条链路：思考开关按厂商方言生成、空正文自动重试与预算放大、
端点拒绝扩展参数时回退、以及各类「带前后缀 / 被截断」的 JSON 容错解析。全部用例不联网即可运行。

---

## 🪪 许可证

[MIT](LICENSE) —— 本仓库仅用于学习交流，请遵守开源协议与当地法律法规。

## 🙏 致谢

- 各开源项目：Gradio、FAISS、sentence-transformers、yt-dlp 等
