"""
视频字幕提取：用 yt-dlp 统一提取多平台长视频字幕。

支持平台（yt-dlp 原生）：YouTube、Bilibili、及其他 yt-dlp 支持的站点。
流程：
  1. DuckDuckGo 限定域名搜索视频 URL（site:youtube.com / site:bilibili.com）
  2. yt-dlp 提取字幕（优先 manual，回退 auto-generated）
  3. 清洗字幕（去时间戳）→ 切分 → 生成 Document
  4. 无字幕的视频静默跳过（不阻断流程）
"""
from __future__ import annotations

import logging
import subprocess
import tempfile
from pathlib import Path

from config import settings, CACHE_DIR
from core.text_processor import clean_subtitle, split_paragraphs
from models.document import Document, SOURCE_VIDEO

logger = logging.getLogger(__name__)

# 字幕语言优先级（与 --sub-langs 保持一致）
_SUB_LANG_PRIORITY = ["zh-Hans", "zh-CN", "zh", "en", "ja", "ru", "es"]


def _subtitle_lang_rank(path: Path) -> int:
    """按语言优先级给字幕文件排序：sub.<lang>.srt 取 <lang> 部分。"""
    lang = path.stem.rsplit(".", 1)[-1] if "." in path.stem else ""
    try:
        return _SUB_LANG_PRIORITY.index(lang)
    except ValueError:
        return len(_SUB_LANG_PRIORITY)


# ==================== 搜索视频 URL ====================

def search_video_urls(
    character_name: str, *, max_per_query: int = 5
) -> list[tuple[str, str]]:
    """
    返回 [(platform, url), ...]
    用 DuckDuckGo 限定视频平台域名搜索。
    """
    try:
        from duckduckgo_search import DDGS
    except ImportError:
        logger.warning("未安装 duckduckgo_search，跳过视频搜索")
        return []

    results: list[tuple[str, str]] = []
    seen: set[str] = set()
    queries = [
        f"{character_name} 纪录片",
        f"{character_name} 讲座 演讲 访谈",
        f"{character_name} 争议 评价 历史分析",       # 多视角内容
        f"{character_name} documentary debate",         # 英文源
    ]

    platforms = settings.search.video_platforms
    try:
        with DDGS() as ddgs:
            for q in queries:
                for platform in platforms:
                    site_q = f"site:{platform} {q}"
                    try:
                        for r in ddgs.text(site_q, max_results=max_per_query):
                            url = r.get("href") or r.get("url") or ""
                            if url and url not in seen:
                                seen.add(url)
                                results.append((platform, url))
                    except Exception as e:
                        logger.debug("视频搜索 %s 失败: %s", site_q, e)
    except Exception as e:
        logger.warning("视频 URL 搜索失败: %s", e)

    return results[: settings.search.video_max_per_query * 2]


# ==================== yt-dlp 提取字幕 ====================

def _extract_subtitle_text(url: str, cache_dir: Path) -> str:
    """
    用 yt-dlp 下载字幕并转为纯文本。
    优先 manual 字幕，回退 auto-generated。
    返回空字符串表示无可用字幕。
    """
    from config import settings
    _proxy = settings.search.proxy
    proxy_args = ["--proxy", _proxy] if _proxy else []  # 空代理 = 直连
    tmp = Path(tempfile.mkdtemp(dir=str(cache_dir), prefix="ytdlp_"))
    try:
        # 1. 列出可用字幕
        list_cmd = [
            "yt-dlp",
            "--skip-download",
            "--list-subs",
            "--no-playlist",
            *proxy_args,
            "--socket-timeout", "15",
            url,
        ]
        try:
            out = subprocess.run(
                list_cmd, capture_output=True, text=True, timeout=60, encoding="utf-8"
            )
            subs_info = out.stdout + out.stderr
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            logger.debug("yt-dlp 列字幕失败 %s: %s", url, e)
            return ""

        # 无任何字幕
        if "has no subtitles" in subs_info or "No subtitles" in subs_info:
            logger.debug("视频无字幕: %s", url)
            return ""

        # 2. 下载字幕（中文优先，回退任意）
        # --write-subs: manual; --write-auto-subs: 自动生成
        out_template = str(tmp / "sub")
        dl_cmd = [
            "yt-dlp",
            "--skip-download",
            "--write-subs",
            "--write-auto-subs",
            "--sub-langs", "zh-Hans,zh,zh-CN,en,ja,ru,es",
            "--sub-format", "vtt/srt/best",
            "--convert-subs", "srt",
            "--no-playlist",
            *proxy_args,
            "--socket-timeout", "15",
            "-o", out_template,
            url,
        ]
        try:
            subprocess.run(dl_cmd, capture_output=True, text=True, timeout=120, encoding="utf-8")
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            logger.debug("yt-dlp 下载字幕失败 %s: %s", url, e)
            return ""

        # 3. 读取生成的字幕文件（按语言优先级排序）
        sub_files = list(tmp.glob("sub*.srt")) + list(tmp.glob("sub*.vtt"))
        if not sub_files:
            return ""
        sub_files.sort(key=_subtitle_lang_rank)
        return sub_files[0].read_text(encoding="utf-8", errors="ignore")
    finally:
        # 清理临时目录
        for f in tmp.glob("*"):
            try:
                f.unlink()
            except OSError:
                pass
        try:
            tmp.rmdir()
        except OSError:
            pass


# ==================== 对外接口 ====================

def collect_video_transcripts(character_name: str) -> list[Document]:
    """
    为某人物采集视频字幕，返回 Document 列表。
    依赖 yt-dlp 命令行工具（pip install yt-dlp）。
    """
    if not settings.search.enable_video_transcript:
        return []

    # 确保 yt-dlp 可用
    try:
        subprocess.run(
            ["yt-dlp", "--version"], capture_output=True, timeout=10, check=True
        )
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        logger.warning("yt-dlp 不可用，跳过视频字幕采集。请确认已 pip install yt-dlp")
        return []

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    video_urls = search_video_urls(character_name)
    if not video_urls:
        logger.info("未搜到 %s 的相关视频", character_name)
        return []

    docs: list[Document] = []
    for platform, url in video_urls:
        raw = _extract_subtitle_text(url, CACHE_DIR)
        if not raw:
            continue
        text = clean_subtitle(raw)
        if len(text) < 50:
            continue
        chunks = split_paragraphs(text)
        for idx, chunk in enumerate(chunks):
            docs.append(
                Document(
                    doc_id=f"video-{character_name}-{platform}-{idx}",
                    character_name=character_name,
                    source_type=SOURCE_VIDEO,
                    source_detail=f"{platform}: {url[:80]}",
                    title=f"{character_name} 视频字幕",
                    language="zh",
                    content=chunk,
                )
            )
        logger.info("视频字幕采集成功: %s (%d 段)", url, len(chunks))

    logger.info("视频字幕为 %s 共采集 %d 段", character_name, len(docs))
    return docs
