"""
后端数据源采集（重写版，按四步链路组织）：

  Step 1. 姓名识别搜索  ——  search_person_candidates(name)
           用 DuckDuckGo + 多语言维基片段确认用户输入的"名字"对应哪位具体人物，
           把片段交给 person_identifier 让 LLM 输出标准身份与三类检索词。

  Step 2. 本人著作/自传 ——  search_person_works(identity)
           按 identity.work_queries 搜 URL，并真正下载网页正文（而非仅搜索摘要），
           兼顾预置著作库、维基文库、Project Gutenberg 等一手文本源。

  Step 3. 他人传记/评传 ——  search_person_biographies(identity)
           按 identity.biography_queries 搜 URL 并抓取正文，
           兼顾多语言维基百科传记段落。

  Step 4. 权威史料/时代背景 ——  search_person_history(identity)
           按 identity.history_queries 搜 URL 并抓取正文，
           涵盖时代背景、官方档案、学术评述。

每步返回 RawSnippet 列表，统一打上 CATEGORY_* 标签，
下游 text_processor.to_documents 负责切分为 Document 段落。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import wikipediaapi

from config import settings
from core.web_fetcher import fetch_url
from models.document import (
    CATEGORY_BIOGRAPHY,
    CATEGORY_HISTORY,
    CATEGORY_WEB,
    CATEGORY_WIKI,
    CATEGORY_WORK,
    SOURCE_WEB,
    SOURCE_WIKI,
    SOURCE_WORK,
)

logger = logging.getLogger(__name__)

# wikipediaapi 必须带 User-Agent，否则部分节点会拒绝
_WIKI_USER_AGENT = "HistoricalAgent/1.0 (educational history dialogue; contact: local)"


def _wiki_kwargs() -> dict:
    """传递给 wikipediaapi.Wikipedia 的 httpx 参数（代理、超时、无重试）。"""
    kwargs: dict = {
        "timeout": settings.search.wiki_timeout,
        # 维基只是补充数据源，站点不可达时应快速失败，不要把单次超时放大 4 倍。
        "max_retries": 0,
        "retry_wait": 0.0,
    }
    proxy = settings.search.proxy
    if proxy:
        kwargs["proxies"] = {"http://": proxy, "https://": proxy}
    return kwargs


@dataclass
class RawSnippet:
    """采集到的原始文本片段（未切分）。

    content 应尽可能为完整正文，而非搜索摘要（与旧版的区别）。
    """
    character_name: str
    source_type: str            # SOURCE_WEB | SOURCE_WIKI | SOURCE_WORK
    source_detail: str          # "DuckDuckGo·著作" | "Wikipedia(zh)" | "《国富论》（预置）"
    title: str
    content: str
    url: str = ""
    language: str = "zh"
    category: str = ""          # 资料类别 CATEGORY_*（work/biography/history/wiki/web）


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


# ==================== DuckDuckGo URL 检索 ====================

def _ddg_search_urls(
    query: str, *, max_results: int
) -> list[tuple[str, str, str]]:
    """
    用 DuckDuckGo 搜索单个 query，返回 [(url, title, snippet), ...]。
    只负责"找到 URL"，不依赖 body 摘要做正文（正文留给 web_fetcher 抓取）。
    """
    try:
        from duckduckgo_search import DDGS
    except ImportError:
        logger.warning("未安装 duckduckgo_search，跳过网页搜索")
        return []

    out: list[tuple[str, str, str]] = []
    try:
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                url = r.get("href") or r.get("url") or ""
                if not url:
                    continue
                title = (r.get("title") or "")[:200]
                snippet = (r.get("body") or "")[:500]
                out.append((url, title, snippet))
    except Exception as e:
        # DuckDuckGo 偶尔限流，不应阻断整个流程
        logger.warning("DuckDuckGo 搜索失败（可能被限流）: %s", e)
    return out


# ==================== 多语言维基百科 ====================

def search_wikipedia(
    character_name: str,
    languages: list[str] | None = None,
    *,
    category: str = CATEGORY_WIKI,
    source_detail_prefix: str = "Wikipedia",
) -> list[RawSnippet]:
    """
    获取多语言维基百科词条。
    先通过中文词条的跨语言链接（langlinks）解析各语言的标准标题，
    避免直接用中文名去英文/俄文 wiki 查询导致词条不存在。
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
        # 中文维基不可达时，仍尝试用原名逐语言抓取，避免丢失资料
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
                    source_detail=f"{source_detail_prefix}({lang})",
                    title=page.title,
                    content=full_text,
                    url=page.fullurl,
                    language=lang,
                    category=category,
                )
            )
            logger.info("维基[%s]获取 %s 词条成功（%d 字）", lang, page.title, len(full_text))
        except Exception as e:
            logger.warning("维基[%s]获取 %s 失败: %s", lang, title, e)

    return snippets


# ==================== Step 1：姓名识别搜索 ====================

def search_person_candidates(
    input_name: str, *, max_results: int = 5
) -> list[RawSnippet]:
    """
    Step 1：轻量级人物识别搜索。
    只用少量网页片段 + 中文/英文维基，快速确认人物身份。
    避免在识别阶段就触发大量深网爬取。
    """
    snippets: list[RawSnippet] = []
    if settings.search.ddg_max_results > 0:
        # 这里只需要搜索摘要片段给 LLM 判断身份，不抓正文
        try:
            from duckduckgo_search import DDGS
            with DDGS() as ddgs:
                for q in [
                    f"{input_name} 是谁 生平",
                    f"{input_name} biography",
                ]:
                    for r in ddgs.text(q, max_results=max_results):
                        url = r.get("href") or r.get("url") or ""
                        snippets.append(
                            RawSnippet(
                                character_name=input_name,
                                source_type=SOURCE_WEB,
                                source_detail="DuckDuckGo·识别",
                                title=(r.get("title") or "")[:200],
                                content=(r.get("body") or "")[:1000],
                                url=url,
                                language="zh",
                                category="",
                            )
                        )
        except Exception as e:
            logger.warning("识别阶段网页搜索失败: %s", e)

    snippets.extend(search_wikipedia(input_name, languages=["zh", "en"]))
    return dedupe_snippets(snippets)


# 兼容旧接口名（person_identifier 仍用）
collect_identity_snippets = search_person_candidates


# ==================== Step 2：本人著作/自传 ====================

def search_person_works(
    identity, *, max_results: int = 5, fetch_fulltext: bool = True
) -> list[RawSnippet]:
    """
    Step 2：搜索并抓取人物**本人著作/自传/作品**全文。

    流程：
      1. 按 identity.work_queries 在 DuckDuckGo 检索 URL；
      2. 对每个 URL 用 web_fetcher 抓取网页正文（修复旧版只拿摘要的短板）；
      3. 抓不到正文时回退到搜索摘要。

    fetch_fulltext=False 时只返回搜索摘要，用于快速识别等场景。
    """
    from core.person_identifier import PersonIdentity

    if not isinstance(identity, PersonIdentity):
        raise TypeError("identity 必须是 PersonIdentity 实例")

    work_queries = identity.work_queries or [
        f"{identity.name} 著作 全文",
        f"{identity.name} works full text",
    ]
    snippets: list[RawSnippet] = []
    seen_urls: set[str] = set()

    for q in work_queries:
        for url, title, snippet in _ddg_search_urls(q, max_results=max_results):
            if url in seen_urls:
                continue
            seen_urls.add(url)

            if fetch_fulltext:
                page = fetch_url(url)
                if page is not None and len(page.text) > len(snippet) * 2:
                    # 正文比摘要长得多，用正文
                    snippets.append(RawSnippet(
                        character_name=identity.name,
                        source_type=SOURCE_WEB,
                        source_detail="DuckDuckGo·著作",
                        title=page.title or title,
                        content=page.text,
                        url=url,
                        language="zh",
                        category=CATEGORY_WORK,
                    ))
                    continue

            # 抓取失败或正文太短，回退到搜索摘要
            snippets.append(RawSnippet(
                character_name=identity.name,
                source_type=SOURCE_WEB,
                source_detail="DuckDuckGo·著作（摘要）",
                title=title,
                content=snippet,
                url=url,
                language="zh",
                category=CATEGORY_WORK,
            ))

    # 补充：多语言维基的"著作"段落（维基词条主文已含作品介绍，这里把维基主词条
    # 也作为著作背景资料一并加入，方便后续做 RAG 时检索作品原文段落）
    wiki = search_wikipedia(
        identity.name,
        languages=["zh", "en"],
        category=CATEGORY_WIKI,
        source_detail_prefix="Wikipedia·著作",
    )
    snippets.extend(wiki)

    logger.info("Step2 本人著作/自传为 %s 采集 %d 条", identity.name, len(snippets))
    return dedupe_snippets(snippets)


# ==================== Step 3：他人传记/评传 ====================

def search_person_biographies(
    identity, *, max_results: int = 5, fetch_fulltext: bool = True
) -> list[RawSnippet]:
    """
    Step 3：搜索并抓取**他人撰写的传记/评传/回忆录**正文。

    与 Step 2 类似的"搜 URL → 抓正文"流程，资料类别标为 biography。
    """
    from core.person_identifier import PersonIdentity

    if not isinstance(identity, PersonIdentity):
        raise TypeError("identity 必须是 PersonIdentity 实例")

    bio_queries = identity.biography_queries or [
        f"{identity.name} 传记",
        f"{identity.name} biography memoir",
    ]
    snippets: list[RawSnippet] = []
    seen_urls: set[str] = set()

    for q in bio_queries:
        for url, title, snippet in _ddg_search_urls(q, max_results=max_results):
            if url in seen_urls:
                continue
            seen_urls.add(url)

            if fetch_fulltext:
                page = fetch_url(url)
                if page is not None and len(page.text) > len(snippet) * 2:
                    snippets.append(RawSnippet(
                        character_name=identity.name,
                        source_type=SOURCE_WEB,
                        source_detail="DuckDuckGo·传记",
                        title=page.title or title,
                        content=page.text,
                        url=url,
                        language="zh",
                        category=CATEGORY_BIOGRAPHY,
                    ))
                    continue

            snippets.append(RawSnippet(
                character_name=identity.name,
                source_type=SOURCE_WEB,
                source_detail="DuckDuckGo·传记（摘要）",
                title=title,
                content=snippet,
                url=url,
                language="zh",
                category=CATEGORY_BIOGRAPHY,
            ))

    logger.info("Step3 他人传记/评传为 %s 采集 %d 条", identity.name, len(snippets))
    return dedupe_snippets(snippets)


# ==================== Step 4：权威史料/时代背景 ====================

def search_person_history(
    identity, *, max_results: int = 5, fetch_fulltext: bool = True
) -> list[RawSnippet]:
    """
    Step 4：搜索并抓取**权威史料/同时代历史背景**正文。

    用于让对话模型理解该人物所处时代的官方档案、历史评述。
    """
    from core.person_identifier import PersonIdentity

    if not isinstance(identity, PersonIdentity):
        raise TypeError("identity 必须是 PersonIdentity 实例")

    history_queries = identity.history_queries or [
        f"{identity.name} 时代背景 历史",
        f"{identity.name} historical background archive",
    ]
    snippets: list[RawSnippet] = []
    seen_urls: set[str] = set()

    for q in history_queries:
        for url, title, snippet in _ddg_search_urls(q, max_results=max_results):
            if url in seen_urls:
                continue
            seen_urls.add(url)

            if fetch_fulltext:
                page = fetch_url(url)
                if page is not None and len(page.text) > len(snippet) * 2:
                    snippets.append(RawSnippet(
                        character_name=identity.name,
                        source_type=SOURCE_WEB,
                        source_detail="DuckDuckGo·史料",
                        title=page.title or title,
                        content=page.text,
                        url=url,
                        language="zh",
                        category=CATEGORY_HISTORY,
                    ))
                    continue

            snippets.append(RawSnippet(
                character_name=identity.name,
                source_type=SOURCE_WEB,
                source_detail="DuckDuckGo·史料（摘要）",
                title=title,
                content=snippet,
                url=url,
                language="zh",
                category=CATEGORY_HISTORY,
            ))

    logger.info("Step4 权威史料/时代背景为 %s 采集 %d 条", identity.name, len(snippets))
    return dedupe_snippets(snippets)


# ==================== 兼容旧接口 ====================

def collect_base_sources(
    character_name: str, *, enable_web: bool = True, enable_wiki: bool = True
) -> list[RawSnippet]:
    """兼容旧调用：未识别身份时直接用名字做基础采集。

    内部走 search_person_candidates + 维基补充。
    """
    snippets: list[RawSnippet] = []
    if enable_web:
        snippets.extend(search_person_candidates(character_name))
    if enable_wiki:
        snippets.extend(
            search_wikipedia(character_name, languages=["zh", "en"])
        )
    return dedupe_snippets(snippets)


def collect_scholar_sources(
    identity,
    *,
    enable_web: bool = True,
    enable_wiki: bool = True,
    max_results: int = 5,
) -> list[RawSnippet]:
    """兼容旧调用：按身份做三类深度资料检索（著作/传记/史料）。

    新链路请直接调用：
      search_person_works / search_person_biographies / search_person_history
    """
    snippets: list[RawSnippet] = []
    if enable_web:
        snippets.extend(search_person_works(identity, max_results=max_results))
        snippets.extend(search_person_biographies(identity, max_results=max_results))
        snippets.extend(search_person_history(identity, max_results=max_results))
    # wiki 已在 search_person_works 内补充，这里不重复
    return dedupe_snippets(snippets)
