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
from dataclasses import dataclass
from pathlib import Path
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

# 单页抓取超时（秒）：包含连接+读取，避免 yt-dlp 之外的环节长时间卡住
_FETCH_TIMEOUT = 20.0
# 单页正文最大字符数：防止某些站点把整本小说塞进来撑爆 token
_MAX_TEXT_CHARS = 20000
# 正文最短字符数：太短的可能是 404/导航页，直接丢弃
_MIN_TEXT_CHARS = 200

# 进程级 URL 缓存（最近 N 条命中即返回），用于同一人物构建期间重复 URL
_URL_CACHE_MAX = 256
_url_cache: dict[str, str] = {}


@dataclass
class FetchedPage:
    """抓取后的网页正文。"""
    url: str
    title: str
    text: str
    status: int = 200


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

    会在 CACHE_DIR 下记录抓取过的原始页面副本，便于离线复现。
    """
    if not url or not url.startswith(("http://", "https://")):
        return None

    # 1. 命中内存缓存直接返回
    cached = _cache_get(url)
    if cached is not None:
        return FetchedPage(url=url, title="", text=cached)

    # 2. 配置代理
    proxy = settings.search.proxy or None
    proxies = {"http://": proxy, "https://": proxy} if proxy else None
    headers = {
        "User-Agent": _DEFAULT_UA,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
    }

    try:
        with httpx.Client(
            timeout=_FETCH_TIMEOUT,
            follow_redirects=True,
            proxies=proxies,
            headers=headers,
        ) as client:
            resp = client.get(url)
    except httpx.HTTPError as e:
        logger.debug("抓取失败 %s: %s", url, e)
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
