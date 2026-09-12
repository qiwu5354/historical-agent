"""
全局配置管理。
支持从环境变量 / .env 文件读取敏感信息（API Key）。
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# .env 固定在项目根目录读取，避免从别的目录启动时读不到配置
_ENV_FILE = Path(__file__).parent.resolve() / ".env"

try:
    from dotenv import load_dotenv
    load_dotenv(_ENV_FILE)
except ImportError:
    pass


# ===== 代理设置 =====
# 依次读取 HTTPS_PROXY / HTTP_PROXY / ALL_PROXY（大小写均可），未配置则为空字符串。
# 空字符串表示「直连」，不强行指定本地代理；但注意 httpx 默认 trust_env=True，
# 因此即使这里为空，系统/环境里已存在的代理变量仍会被底层库使用。
_PROXY_ENV_KEYS = ("HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy", "ALL_PROXY", "all_proxy")


def _resolve_proxy() -> str:
    """从环境变量解析采集用代理地址；未配置返回空字符串。"""
    for key in _PROXY_ENV_KEYS:
        value = (os.getenv(key) or "").strip()
        if value:
            return value
    return ""


_PROXY = _resolve_proxy()


def _env_float(key: str, default: float) -> float:
    """读取浮点环境变量，非法值回退默认值，避免启动时崩溃。"""
    try:
        return float(os.getenv(key, str(default)))
    except (TypeError, ValueError):
        return default


def _env_int(key: str, default: int) -> int:
    """读取整数环境变量，非法值回退默认值，避免启动时崩溃。"""
    try:
        return int(os.getenv(key, str(default)))
    except (TypeError, ValueError):
        return default


# ===== 路径配置 =====
BASE_DIR = Path(__file__).parent.resolve()
DATA_DIR = BASE_DIR / "data"
DB_PATH = DATA_DIR / "historical_agent.db"
FAISS_DIR = DATA_DIR / "faiss_index"
CACHE_DIR = DATA_DIR / "cache"
ASSETS_DIR = BASE_DIR / "assets"
WORKS_DIR = ASSETS_DIR / "works"


@dataclass
class LLMConfig:
    """LLM 接入配置（OpenAI 兼容接口，可接智谱/通义/DeepSeek等）"""
    # 优先从环境变量读取，默认值为通义千问（DashScope）示例
    api_key: str = field(default_factory=lambda: os.getenv("LLM_API_KEY", ""))
    base_url: str = field(default_factory=lambda: os.getenv(
        "LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
    ))
    model: str = field(default_factory=lambda: os.getenv("LLM_MODEL", "qwen3-turbo"))
    temperature: float = field(default_factory=lambda: _env_float("LLM_TEMPERATURE", 0.7))
    # 单次回复的最大 token 预算。注意：混合思考模型的「思考」与「正文」共享该预算，
    # 2048 在长档案提取场景下容易被思考吃光导致正文为空，故默认给到 4096。
    max_tokens: int = field(default_factory=lambda: _env_int("LLM_MAX_TOKENS", 4096))
    # 是否关闭「思考模式」。qwen3.5+/qwen3.7-flash、GLM-5 等默认开启思考，
    # 思考 token 会挤占 max_tokens 导致正文被截断甚至为空，故默认关闭。
    # 想要更高质量推理可设为 0，但同时务必把 LLM_MAX_TOKENS 调到 8192 以上。
    disable_thinking: bool = field(
        default_factory=lambda: os.getenv("LLM_DISABLE_THINKING", "1") != "0"
    )
    # 用于档案提取/翻译等"严肃"任务的低温度模型
    serious_temperature: float = 0.2
    # 流式输出的 chunk 超时（秒）
    stream_timeout: float = 120.0


@dataclass
class SearchConfig:
    """数据源采集配置"""
    # 代理（访问维基/DDG 等海外站点需要；空字符串 = 直连，交给 httpx trust_env 决定）
    proxy: str = field(default_factory=_resolve_proxy)
    # DuckDuckGo
    ddg_max_results: int = 10
    # Wikipedia 多语言：人物母语映射（值是维基语言代码）
    wiki_languages: list[str] = field(
        default_factory=lambda: ["zh", "en", "ru", "es"]
    )
    # 视频字幕
    enable_video_transcript: bool = True
    video_max_per_query: int = 3       # 每次搜索最多提取几个视频字幕
    video_platforms: list[str] = field(
        default_factory=lambda: ["youtube.com", "bilibili.com"]
    )
    # wikipedia（wikipediaapi 内部自带重试次数，这里为单次请求超时；站点不可达时快速跳过）
    wiki_timeout: float = 8.0          # 单次请求超时（秒）
    wiki_max_retries: int = 0          # 传输错误重试次数（0 = 快速失败，避免单次放大 4 倍）
    # 网页抓取：单页超时与重试
    fetch_timeout: float = 20.0
    fetch_retries: int = 1             # 传输错误重试次数（首次之外再试几次）


@dataclass
class RAGConfig:
    """向量检索配置"""
    # 多语言向量模型（中文友好、轻量）
    embedding_model: str = "paraphrase-multilingual-MiniLM-L12-v2"
    # 文档切分
    chunk_size: int = 500              # 每段字符数
    chunk_overlap: int = 50
    # 检索
    top_k: int = 5                     # 每次检索返回段落数


@dataclass
class Settings:
    llm: LLMConfig = field(default_factory=LLMConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    rag: RAGConfig = field(default_factory=RAGConfig)
    debug: bool = os.getenv("DEBUG", "0") == "1"


# 全局单例
settings = Settings()


def ensure_dirs() -> None:
    """启动时确保运行时目录存在"""
    for d in (DATA_DIR, FAISS_DIR, CACHE_DIR):
        d.mkdir(parents=True, exist_ok=True)
