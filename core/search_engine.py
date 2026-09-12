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

import httpx
import wikipediaapi

from config import settings
from core.web_fetcher import _normalize_proxy, fetch_url, http_client_config, proxy_hint
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
    """
    传递给 wikipediaapi.Wikipedia 的参数。

    wikipediaapi 会把未知 kwargs 直接转发给 httpx.Client，因此这里**只能**放
    httpx 认识的参数；旧代码传的 `proxies={...}` 在 httpx ≥ 0.28 上会抛
    `TypeError: unexpected keyword argument 'proxies'`，导致维基采集全线失败。
    正确写法是 `proxy="http://host:port"`（单个地址字符串）。
    `max_retries` / `retry_wait` 是 wikipediaapi 自己的参数，仍然可用。
    """
    kwargs: dict = {
        "timeout": settings.search.wiki_timeout,
        # 维基只是补充数据源，站点不可达时应快速失败，不要把单次超时放大 4 倍。
        "max_retries": settings.search.wiki_max_retries,
        "retry_wait": 0.0,
    }
    proxy = _normalize_proxy(settings.search.proxy)
    if proxy:
        kwargs["proxy"] = proxy
    return kwargs


# ==================== 维基 REST 摘要兜底（不依赖 wikipediaapi）====================

def _quote(title: str) -> str:
    from urllib.parse import quote

    return quote(title.replace(" ", "_"), safe="")


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


def _wiki_rest_summary(
    lang: str, title: str, *, category: str, source_detail: str,
    character_name: str,
) -> list[RawSnippet]:
    """
    用 MediaWiki REST API 直接取词条摘要，作为 wikipediaapi 失败时的兜底。

    背景：wikipediaapi 走的是 `action=query` 接口，部分网络环境只放行
    `/api/rest_v1/`；多一条路径就多一次拿到资料的机会。失败静默返回空列表。
    """
    if not title:
        return []
    url = f"https://{lang}.wikipedia.org/api/rest_v1/page/summary/{_quote(title)}"
    try:
        with httpx.Client(**http_client_config(timeout=settings.search.wiki_timeout)) as client:
            resp = client.get(url)
        if resp.status_code != 200:
            logger.debug("维基 REST[%s] %s 返回 %d", lang, title, resp.status_code)
            return []
        data = resp.json()
    except Exception as e:  # noqa: BLE001
        logger.debug("维基 REST[%s] %s 失败: %s", lang, title, e)
        return []

    extract = (data.get("extract") or "").strip()
    if len(extract) < 50:
        return []
    page_title = data.get("title") or title
    content = f"{data.get('description') or ''}\n\n{extract}".strip()
    urls = data.get("content_urls") or {}
    full_url = ((urls.get("desktop") or {}).get("page")) or url
    logger.info("维基REST[%s]获取 %s 摘要成功（%d 字）", lang, page_title, len(content))
    return [
        RawSnippet(
            character_name=character_name,
            source_type=SOURCE_WIKI,
            source_detail=f"{source_detail}({lang}·REST)",
            title=page_title,
            content=content,
            url=full_url,
            language=lang,
            category=category,
        )
    ]


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

    兼容两代包名：`duckduckgo_search` 已改名为 `ddgs`，优先用新包以避开弃用告警。
    """
    DDGS = _import_ddgs()
    if DDGS is None:
        logger.warning("未安装 duckduckgo_search / ddgs，跳过 DuckDuckGo 搜索")
        return []

    proxy = _ddg_proxy()
    out: list[tuple[str, str, str]] = []
    try:
        with DDGS(proxy=proxy) as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                url = r.get("href") or r.get("url") or ""
                if not url:
                    continue
                title = (r.get("title") or "")[:200]
                snippet = (r.get("body") or "")[:1200]
                out.append((url, title, snippet))
    except Exception as e:
        # DuckDuckGo 偶尔限流，不应阻断整个流程
        logger.warning("DuckDuckGo 搜索失败（可能被限流/网络不可达）: %s", e)
    return out


def _import_ddgs():
    """返回可用的 DDGS 类（优先新包 ddgs，回退旧包 duckduckgo_search）。"""
    try:
        from ddgs import DDGS  # type: ignore[import-not-found]

        return DDGS
    except ImportError:
        pass
    try:
        from duckduckgo_search import DDGS

        return DDGS
    except ImportError:
        return None


# SOCKS 代理告警只打印一次，避免每个 query 都刷屏
_socks_warned = False


def _ddg_proxy() -> str | None:
    """
    解析 DuckDuckGo 要用的代理。

    DDGS 只支持 HTTP(S) 代理；传 `socks5://` 会抛 ValueError 让搜索全面失败，
    因此这里对 SOCKS 代理直接跳过（返回 None 并告警一次），交由 Bing RSS 兜底。
    """
    global _socks_warned
    proxy = _normalize_proxy(settings.search.proxy)
    if not proxy:
        return None
    if proxy.startswith(("socks4", "socks5", "socks")):
        if not _socks_warned:
            _socks_warned = True
            logger.warning(
                "DuckDuckGo 客户端不支持 SOCKS 代理（%s），本次跳过 DDG，改用 Bing RSS 兜底。"
                "若想启用 DDG，请把 HTTPS_PROXY 改为 http:// 形式的代理地址。",
                proxy,
            )
        return None
    return proxy



def _bing_rss_search_urls(
    query: str, *, max_results: int
) -> list[tuple[str, str, str]]:
    """
    Bing 新闻/网页 RSS 兜底搜索，返回与 _ddg_search_urls 相同的结构。

    为什么需要它：DuckDuckGo 在国内网络常年不可达或限流，一旦它返回空，
    整条「搜索 → 抓正文」的链路就全空（表现为「共 0 段资料」）。
    Bing RSS 走普通 HTTPS，能走同一个代理，作为第二检索源显著提高命中率。
    """
    from urllib.parse import quote_plus
    from xml.etree import ElementTree

    url = f"https://www.bing.com/search?q={quote_plus(query)}&format=rss&count={max_results}"
    try:
        with httpx.Client(**http_client_config(timeout=15.0)) as client:
            resp = client.get(url)
        if resp.status_code != 200:
            logger.debug("Bing RSS 返回 %d", resp.status_code)
            return []
        root = ElementTree.fromstring(resp.content)
    except Exception as e:  # noqa: BLE001
        logger.debug("Bing RSS 搜索失败: %s", e)
        return []

    out: list[tuple[str, str, str]] = []
    for item in root.iter("item"):
        link = (item.findtext("link") or "").strip()
        title = (item.findtext("title") or "").strip()[:200]
        desc = (item.findtext("description") or "").strip()[:1200]
        if link.startswith(("http://", "https://")):
            out.append((link, title, desc))
        if len(out) >= max_results:
            break
    if out:
        logger.info("Bing RSS 命中 %d 条（DuckDuckGo 不可用时的兜底源）", len(out))
    return out


def _web_search_urls(query: str, *, max_results: int) -> list[tuple[str, str, str]]:
    """统一网页检索入口：DuckDuckGo 优先，空结果时用 Bing RSS 兜底。"""
    results = _ddg_search_urls(query, max_results=max_results)
    if results:
        return results
    return _bing_rss_search_urls(query, max_results=max_results)


def web_search_health() -> tuple[bool, str]:
    """探测网页检索是否可用（供 data_collector.health_check 汇总）。"""
    try:
        rows = _web_search_urls("维基百科", max_results=2)
    except Exception as e:  # noqa: BLE001
        return False, f"网页检索异常: {type(e).__name__}: {e}"
    if rows:
        return True, f"网页检索可用（返回 {len(rows)} 条）"
    return False, "DuckDuckGo 与 Bing 均无结果（网络不可达或均被限流）"



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
    # 记录 wikipediaapi 彻底不可用的语言（连接/超时），这些语言改走 REST 兜底
    failed_langs: list[str] = []

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
        failed_langs.append("zh")

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
            failed_langs.append(lang)

    # 3. REST 兜底：wikipediaapi（action=query）不可达时，改用 /api/rest_v1/ 摘要接口。
    #    有些网络环境只放行后者，多一条路径就多一次拿到资料的机会。
    got_langs = {s.language for s in snippets}
    for lang in dict.fromkeys(failed_langs):
        if lang in got_langs:
            continue
        fallback = _wiki_rest_summary(
            lang,
            titles.get(lang, character_name),
            category=category,
            source_detail=source_detail_prefix,
            character_name=character_name,
        )
        snippets.extend(fallback)

    if snippets:
        logger.info("维基采集成功：%d 条（%s）", len(snippets), "、".join(sorted({s.language for s in snippets})))
    else:
        logger.warning(
            "维基百科未取到任何词条（已尝试语言：%s）。%s",
            "、".join(langs), proxy_hint(),
        )

    return snippets


def wikipedia_health() -> tuple[bool, str]:
    """探测维基百科是否可用（wikipediaapi 与 REST 两条路径任一通即算可用）。"""
    try:
        page = wikipediaapi.Wikipedia(
            language="zh",
            user_agent=_WIKI_USER_AGENT,
            extract_format=wikipediaapi.ExtractFormat.WIKI,
            **_wiki_kwargs(),
        ).page("马克思")
        if page.exists() and len(page.summary) > 50:
            return True, f"wikipediaapi 可用（zh 词条 {page.title}）"
    except Exception as e:  # noqa: BLE001
        logger.debug("wikipediaapi 探测失败: %s", e)

    if _wiki_rest_summary(
        "zh", "马克思", category=CATEGORY_WIKI,
        source_detail="Wikipedia·健康检查", character_name="马克思",
    ):
        return True, "wikipediaapi 不可用，但 REST 摘要接口可用"

    return False, f"维基百科不可达（wikipediaapi 与 REST 均失败）。{proxy_hint()}"



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
        for q in [
            f"{input_name} 是谁 生平",
            f"{input_name} biography",
        ]:
            for url, title, snippet in _web_search_urls(q, max_results=max_results):
                snippets.append(
                    RawSnippet(
                        character_name=input_name,
                        source_type=SOURCE_WEB,
                        source_detail="网页检索·识别",
                        title=title,
                        content=snippet[:1000],
                        url=url,
                        language="zh",
                        category="",
                    )
                )

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
        for url, title, snippet in _web_search_urls(q, max_results=max_results):
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
                        source_detail="网页检索·著作",
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
                source_detail="网页检索·著作（搜索摘要）",
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
        for url, title, snippet in _web_search_urls(q, max_results=max_results):
            if url in seen_urls:
                continue
            seen_urls.add(url)

            if fetch_fulltext:
                page = fetch_url(url)
                if page is not None and len(page.text) > len(snippet) * 2:
                    snippets.append(RawSnippet(
                        character_name=identity.name,
                        source_type=SOURCE_WEB,
                        source_detail="网页检索·传记",
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
                source_detail="网页检索·传记（搜索摘要）",
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
        for url, title, snippet in _web_search_urls(q, max_results=max_results):
            if url in seen_urls:
                continue
            seen_urls.add(url)

            if fetch_fulltext:
                page = fetch_url(url)
                if page is not None and len(page.text) > len(snippet) * 2:
                    snippets.append(RawSnippet(
                        character_name=identity.name,
                        source_type=SOURCE_WEB,
                        source_detail="网页检索·史料",
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
                source_detail="网页检索·史料（搜索摘要）",
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
