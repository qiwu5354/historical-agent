"""
基础数据源采集：DuckDuckGo 搜索 + 多语言维基百科。

只负责"采集原始文本片段"，不做切分入库（那是 text_processor 的职责）。
返回统一的 RawSnippet 结构，附带来源信息。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import wikipediaapi  # 中文友好，支持多语言

from config import settings
from models.document import SOURCE_WEB, SOURCE_WIKI

logger = logging.getLogger(__name__)

# wikipediaapi 要求带 User-Agent，否则部分节点会拒绝
_WIKI_USER_AGENT = "HistoricalAgent/1.0 (educational history dialogue; contact: local)"

# 构建 wikipediaapi 的额外 httpx 参数（代理、超时）
def _wiki_kwargs() -> dict:
    """返回传递给 wikipediaapi.Wikipedia 的 extra kwargs（httpx 参数）。"""
    kwargs: dict = {
        "timeout": settings.search.wiki_timeout,
        # 关闭内置重试：维基只是补充数据源，站点不可达时应快速失败，
        # 否则默认 max_retries=3 会把单次超时放大 4 倍，严重拖慢构建。
        "max_retries": 0,
        "retry_wait": 0.0,
    }
    proxy = settings.search.proxy
    if proxy:
        # httpx 代理格式：{"http://": proxy, "https://": proxy}
        kwargs["proxies"] = {"http://": proxy, "https://": proxy}
    return kwargs


@dataclass
class RawSnippet:
    """采集到的原始文本片段（未切分）"""
    character_name: str
    source_type: str            # SOURCE_WEB | SOURCE_WIKI
    source_detail: str          # "DuckDuckGo" | "Wikipedia(zh)" | "Wikipedia(es)"
    title: str
    content: str
    url: str = ""
    language: str = "zh"



def format_raw_snippets(snippets: list[RawSnippet], limit: int = 20) -> str:
    """把检索片段压缩成给 LLM 识别人物用的文本。"""
    blocks: list[str] = []
    for s in snippets[:limit]:
        blocks.append(
            f"[{s.source_detail}] {s.title}\n{s.content[:500]}"
        )
    return "\n\n---\n\n".join(blocks)


def dedupe_snippets(snippets: list[RawSnippet]) -> list[RawSnippet]:
    """按 URL / 标题 / 内容首段去重，避免同一资料重复进入知识库。"""
    seen: set[str] = set()
    result: list[RawSnippet] = []
    for s in snippets:
        key = s.url or s.title or s.content[:80]
        if key in seen:
            continue
        seen.add(key)
        result.append(s)
    return result


# ==================== DuckDuckGo 搜索 ====================

def search_web(
    character_name: str,
    *,
    max_results: int | None = None,
    queries: list[str] | None = None,
    source_detail: str = "DuckDuckGo",
) -> list[RawSnippet]:
    """用 DuckDuckGo 搜索人物相关网页摘要。

    queries 为空时使用默认通用查询；否则按传入查询逐条搜索。
    source_detail 可用于标记“著作/传记/史料”等来源类型。
    """
    max_results = max_results or settings.search.ddg_max_results
    snippets: list[RawSnippet] = []
    try:
        from duckduckgo_search import DDGS
    except ImportError:
        logger.warning("未安装 duckduckgo_search，跳过网页搜索")
        return snippets

    if queries is None:
        queries = [
            f"{character_name} 生平 思想 贡献",
            f"{character_name} 理论 著作 历史评价",
            f"{character_name} 争议 批评 局限性",         # 批判性视角
            f"{character_name} biography thought legacy",  # 英文/国际视角
            f"{character_name} criticism controversy",     # 英文批判视角
        ]
    seen_urls: set[str] = set()
    try:
        with DDGS() as ddgs:
            for q in queries:
                results = ddgs.text(q, max_results=max_results)
                for r in results:
                    url = r.get("href") or r.get("url") or ""
                    if url in seen_urls:
                        continue
                    seen_urls.add(url)
                    snippets.append(
                        RawSnippet(
                            character_name=character_name,
                            source_type=SOURCE_WEB,
                            source_detail=source_detail,
                            title=r.get("title", "")[:200],
                            content=r.get("body", "")[:1000],
                            url=url,
                        )
                    )
    except Exception as e:
        # DuckDuckGo 偶尔限流，不应阻断整个流程
        logger.warning("DuckDuckGo 搜索失败（可能被限流）: %s", e)

    logger.info("DuckDuckGo 为 %s 采集到 %d 条片段", character_name, len(snippets))
    return snippets


# ==================== 多语言维基百科 ====================

def search_wikipedia(
    character_name: str, languages: list[str] | None = None
) -> list[RawSnippet]:
    """
    获取多语言维基百科词条。
    先通过中文词条的跨语言链接（langlinks）解析各语言的标准标题，
    避免直接用中文名去英文/俄文/西班牙文等 wiki 查询导致词条不存在。
    """
    langs = languages or settings.search.wiki_languages
    snippets: list[RawSnippet] = []

    def _make_wiki(lang: str) -> wikipediaapi.Wikipedia:
        return wikipediaapi.Wikipedia(
            language=lang,
            user_agent=_WIKI_USER_AGENT,
            extract_format=wikipediaapi.ExtractFormat.WIKI,
            **_wiki_kwargs(),
        )

    # 1. 通过中文词条的 langlinks 解析各语言标准标题
    titles: dict[str, str] = {}
    try:
        zh_page = _make_wiki("zh").page(character_name)
        if zh_page.exists():
            titles["zh"] = zh_page.title
            for lang, link in zh_page.langlinks.items():
                titles.setdefault(lang, link.title)
    except Exception as e:
        # 中文维基不可达时，不直接放弃；仍尝试用原名逐语言抓取，
        # 避免中文维基被墙但英文/其他语言维基可用时丢失资料。
        logger.warning("中文维基访问失败，将尝试用原名逐语言抓取: %s", e)
        titles = {}

    # 2. 逐语言抓取
    for lang in langs:
        title = titles.get(lang, character_name)
        try:
            page = _make_wiki(lang).page(title)
            if not page.exists():
                logger.debug("维基[%s]无 %s 词条", lang, title)
                continue
            summary = page.summary
            if len(summary) < 50:
                continue
            full_text = page.text[:8000] if page.text else summary
            snippets.append(
                RawSnippet(
                    character_name=character_name,
                    source_type=SOURCE_WIKI,
                    source_detail=f"Wikipedia({lang})",
                    title=page.title,
                    content=full_text,
                    url=page.fullurl,
                    language=lang,
                )
            )
            logger.info("维基[%s]获取 %s 词条成功（%d 字）", lang, page.title, len(full_text))
        except Exception as e:
            logger.warning("维基[%s]获取 %s 失败: %s", lang, title, e)

    return snippets


# ==================== 统一入口 ====================

def collect_base_sources(
    character_name: str, *, enable_web: bool = True, enable_wiki: bool = True
) -> list[RawSnippet]:
    """
    采集基础数据源（网页 + 维基）。
    """
    snippets: list[RawSnippet] = []
    if enable_web:
        snippets.extend(search_web(character_name))
    if enable_wiki:
        snippets.extend(search_wikipedia(character_name))
    logger.info("基础数据源为 %s 共采集 %d 条片段", character_name, len(snippets))
    return snippets



def collect_identity_snippets(
    input_name: str, *, max_results: int = 5
) -> list[RawSnippet]:
    """
    轻量级人物识别检索：只用少量网页片段 + 中文/英文维基，快速确认人物身份。
    避免在识别阶段就触发大量深网爬取。
    """
    snippets: list[RawSnippet] = []
    if settings.search.ddg_max_results > 0:
        snippets.extend(
            search_web(
                input_name,
                max_results=max_results,
                queries=[
                    f"{input_name} 是谁 生平",
                    f"{input_name} biography",
                ],
                source_detail="DuckDuckGo·识别",
            )
        )
    snippets.extend(search_wikipedia(input_name, languages=["zh", "en"]))
    return dedupe_snippets(snippets)


def collect_scholar_sources(
    identity,
    *,
    enable_web: bool = True,
    enable_wiki: bool = True,
    max_results: int = 5,
) -> list[RawSnippet]:
    """
    深度资料检索：按人物身份生成的三类查询词检索：
      1. 本人著作/作品
      2. 传记/评传/回忆录
      3. 历史阶段权威史料/背景
    并补充多语言维基。
    """
    from core.person_identifier import PersonIdentity

    if not isinstance(identity, PersonIdentity):
        raise TypeError("identity 必须是 PersonIdentity 实例")

    snippets: list[RawSnippet] = []
    if enable_web:
        work_queries = identity.work_queries or [f"{identity.name} 著作 全文"]
        bio_queries = identity.biography_queries or [f"{identity.name} 传记"]
        history_queries = identity.history_queries or [f"{identity.name} 历史背景 档案"]

        # 分别搜索，并给来源打上“著作/传记/史料”标签，便于 RAG 溯源
        for q in work_queries:
            snippets.extend(
                search_web(
                    identity.name,
                    max_results=max_results,
                    queries=[q],
                    source_detail="DuckDuckGo·著作",
                )
            )
        for q in bio_queries:
            snippets.extend(
                search_web(
                    identity.name,
                    max_results=max_results,
                    queries=[q],
                    source_detail="DuckDuckGo·传记",
                )
            )
        for q in history_queries:
            snippets.extend(
                search_web(
                    identity.name,
                    max_results=max_results,
                    queries=[q],
                    source_detail="DuckDuckGo·史料",
                )
            )

    if enable_wiki:
        snippets.extend(search_wikipedia(identity.name, languages=["zh", "en"]))

    return dedupe_snippets(snippets)
