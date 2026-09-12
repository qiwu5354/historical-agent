"""
网页正文抓取器：把 DuckDuckGo 搜到的 URL 真正下载下来并提取正文文本，
而非仅依赖搜索结果摘要（旧版只能拿到 ~1000 字摘要，不足以做深度 RAG）。

设计要点：
  - 用 httpx 同步 client，单次请求带超时与 UA；
  - 用 BeautifulSoup + lxml 提取 <article> / <main> / 段落文本，去掉脚本/导航/页脚；
  - 全站失败不阻断主流程，调用方按 try/except 处理；
  - 维护一个进程级 LRU 缓存（按 URL SHA-1），避免短时间内重复抓取同一页面。
"""
from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass
from typing import Optional

import httpx

from config import CACHE_DIR, settings

logger = logging.getLogger(__name__)

# 默认 User-Agent：很多站点拒绝默认 python-httpx UA
_DEFAULT_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)

# 单页正文最大字符数：防止某些站点把整本小说塞进来撑爆 token
_MAX_TEXT_CHARS = 20000
# 正文最短字符数：太短的可能是 404/导航页，直接丢弃
_MIN_TEXT_CHARS = 200
# 传输类错误退避基数（秒），仅用于重试之间的短暂等待
_RETRY_BACKOFF = 0.6

# 进程级 URL 缓存（最近 N 条命中即返回），用于同一人物构建期间重复 URL
_URL_CACHE_MAX = 256
_url_cache: dict[str, str] = {}

# 只对「网络传输类」错误重试；HTTP 4xx/5xx 属于服务端明确答复，重试无意义
_RETRYABLE_ERRORS = (httpx.TimeoutException, httpx.ConnectError, httpx.NetworkError)


@dataclass
class FetchedPage:
    """抓取后的网页正文。"""
    url: str
    title: str
    text: str
    status: int = 200


def _normalize_proxy(proxy: str | None) -> str | None:
    """
    规范化代理地址：允许只写 `127.0.0.1:7890`（补全 http://）。
    httpx 要求带 scheme，否则每次请求都会抛「Invalid URL」，且报错信息很难定位。
    """
    proxy = (proxy or "").strip()
    if not proxy:
        return None
    if "://" not in proxy:
        proxy = f"http://{proxy}"
    return proxy


def http_client_config(*, timeout: float | None = None) -> dict:
    """
    统一的 httpx 客户端参数（web_fetcher / search_engine 共用），版本无关。

    重要：httpx ≥ 0.28 已移除 `proxies=` 参数，只能用 `proxy=`（单个地址）。
    旧代码写 `httpx.Client(proxies=...)` 会抛
    `TypeError: Client.__init__() got an unexpected keyword argument 'proxies'`，
    该异常不是 httpx.HTTPError、不会被捕获，导致整条采集链路静默失败
    （表现为「共 0 段资料」）。
    """
    cfg = settings.search
    config: dict = {
        "timeout": cfg.fetch_timeout if timeout is None else timeout,
        "follow_redirects": True,
        # 让已导出的 HTTP(S)_PROXY 也能生效（显式 proxy= 时 httpx 会自动忽略 trust_env）
        "trust_env": True,
        "headers": {
            "User-Agent": _DEFAULT_UA,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        },
    }
    proxy = _normalize_proxy(cfg.proxy)
    if proxy:
        config["proxy"] = proxy
    return config


def proxy_hint() -> str:
    """代理相关的一句诊断提示，用于失败日志。"""
    proxy = _normalize_proxy(settings.search.proxy)
    if proxy:
        return (
            f"当前代理：{proxy}（若该代理未启动，所有海外站点都会失败；"
            "请在 .env 中把 HTTPS_PROXY 改成实际可用的地址，或清空以直连）"
        )
    return (
        "当前未配置代理（直连）。若本机无法直连维基/DuckDuckGo，"
        "请在 .env 中设置 HTTPS_PROXY=http://127.0.0.1:7890（端口按你的代理软件填）"
    )



def _cache_key(url: str) -> str:
    return hashlib.sha1(url.encode("utf-8")).hexdigest()


def _cache_get(url: str) -> Optional[str]:
    """命中缓存则返回正文，否则 None。简单 LRU：超容量时按插入顺序删除首个。"""
    key = _cache_key(url)
    text = _url_cache.get(key)
    if text is not None:
        # 简单 LRU：重新插入以维持顺序
        _url_cache.pop(key, None)
        _url_cache[key] = text
        return text
    return None


def _cache_put(url: str, text: str) -> None:
    if len(_url_cache) >= _URL_CACHE_MAX:
        # popitem(last=False) 删除最旧
        try:
            _url_cache.pop(next(iter(_url_cache)))
        except StopIteration:
            pass
    _url_cache[_cache_key(url)] = text


def _extract_text(html: str) -> tuple[str, str]:
    """从 HTML 提取 (title, 正文)。优先 <article>/<main>，回退全 <p>。"""
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        # 无 bs4 时退化到极简正则提取
        import re
        title_match = re.search(r"<title[^>]*>([^<]*)</title>", html, re.IGNORECASE)
        title = title_match.group(1).strip() if title_match else ""
        # 去标签
        text = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return title, text

    soup = BeautifulSoup(html, "lxml")

    # 提取标题
    title = ""
    if soup.title and soup.title.string:
        title = soup.title.string.strip()

    # 移除明显的噪声节点
    for tag in soup(["script", "style", "noscript", "nav", "header", "footer", "aside", "form"]):
        tag.decompose()

    # 优先 <article> / <main>，再回退到全 <p>
    body = soup.find("article") or soup.find("main") or soup
    paragraphs = body.find_all("p") if body is not None else []
    text_parts: list[str] = []
    for p in paragraphs:
        t = p.get_text(separator=" ", strip=True)
        if t:
            text_parts.append(t)
    text = "\n\n".join(text_parts)

    # 若 <p> 太少（可能是非文章页），回退到 body 全文
    if len(text) < _MIN_TEXT_CHARS and body is not None:
        text = body.get_text(separator="\n", strip=True)

    return title, text


def fetch_url(url: str, *, max_chars: int = _MAX_TEXT_CHARS) -> Optional[FetchedPage]:
    """
    抓取单个 URL 的正文。失败返回 None，不抛异常（由调用方决定是否记日志）。

    传输类错误（超时/连接失败）按 settings.search.fetch_retries 短暂重试；
    会在 CACHE_DIR 下记录抓取过的原始页面副本，便于离线复现。
    """
    if not url or not url.startswith(("http://", "https://")):
        return None

    # 1. 命中内存缓存直接返回
    cached = _cache_get(url)
    if cached is not None:
        return FetchedPage(url=url, title="", text=cached)

    tries = max(1, int(settings.search.fetch_retries) + 1)
    resp: httpx.Response | None = None
    last_error: Exception | None = None

    for attempt in range(tries):
        try:
            with httpx.Client(**http_client_config()) as client:
                resp = client.get(url)
            break
        except _RETRYABLE_ERRORS as e:
            last_error = e
            if attempt + 1 < tries:
                time.sleep(_RETRY_BACKOFF * (attempt + 1))
                continue
            logger.debug("抓取失败 %s: %s（%s）", url, e, proxy_hint())
            return None
        except Exception as e:  # noqa: BLE001
            # 非传输类错误（如参数不兼容、无效代理地址）必须显式记警告：
            # 旧版把这类错误漏出 except httpx.HTTPError 之外，导致整条链路静默失败。
            logger.warning("抓取 %s 出现非预期错误: %s: %s", url, type(e).__name__, e)
            if attempt + 1 < tries:
                time.sleep(_RETRY_BACKOFF * (attempt + 1))
                continue
            return None

    if resp is None:
        logger.debug("抓取 %s 失败: %s", url, last_error)
        return None

    if resp.status_code != 200:
        logger.debug("抓取 %s 返回 %d", url, resp.status_code)
        return None

    content_type = resp.headers.get("content-type", "")
    if "text" not in content_type and "html" not in content_type and "xml" not in content_type:
        # 非 HTML/文本（如 PDF/图片），跳过
        return None

    # 解码：httpx 默认会处理，兜底用 utf-8
    try:
        html = resp.content.decode(resp.encoding or "utf-8", errors="ignore")
    except (LookupError, UnicodeDecodeError):
        html = resp.content.decode("utf-8", errors="ignore")

    title, text = _extract_text(html)
    text = text.strip()

    if len(text) < _MIN_TEXT_CHARS:
        # 正文过短，可能是导航页/404，丢弃
        return None

    if len(text) > max_chars:
        text = text[:max_chars] + "\n\n[...正文过长，已截断...]"

    # 缓存：内存 + 磁盘（便于离线复现，磁盘只存原文）
    _cache_put(url, text)
    try:
        disk_path = CACHE_DIR / "web_fetch"
        disk_path.mkdir(parents=True, exist_ok=True)
        (disk_path / f"{_cache_key(url)}.txt").write_text(
            f"URL: {url}\nTITLE: {title}\n\n{text}", encoding="utf-8"
        )
    except OSError:
        pass  # 磁盘写入失败不影响主流程

    return FetchedPage(url=url, title=title or url, text=text, status=resp.status_code)


def fetch_urls(
    urls: list[str], *, max_chars: int = _MAX_TEXT_CHARS
) -> list[FetchedPage]:
    """批量抓取多个 URL。任一失败不影响其他。"""
    out: list[FetchedPage] = []
    for url in urls:
        page = fetch_url(url, max_chars=max_chars)
        if page is not None:
            out.append(page)
    return out


def reset_cache() -> None:
    """清空进程内 URL 缓存（供测试或重新构建时调用）。"""
    _url_cache.clear()
