"""
统一数据采集器（重写版，按四步链路串联）。

新数据流：
  Step 0. 预置著作库（离线高质量文本，若已注册）
  Step 1. 姓名识别搜索 → search_person_candidates
          （识别阶段在 character_builder 中已完成，这里只承担资料采集）
  Step 2. 本人著作/自传 → search_person_works
  Step 3. 他人传记/评传 → search_person_biographies
  Step 4. 权威史料/时代背景 → search_person_history
  Step 5. 可选视频字幕 → collect_video_transcripts

所有源的原始文本都经过 text_processor.to_documents 切分为 Document 段落，
统一打上 CATEGORY_* 标签，便于下游 RAG 溯源。
"""
from __future__ import annotations

import hashlib
import logging

from config import WORKS_DIR
from core.person_identifier import PersonIdentity
from core.search_engine import (
    RawSnippet,
    search_person_biographies,
    search_person_history,
    search_person_works,
)
from core.text_processor import to_documents, normalize_text, clean_html
from core.video_transcript import collect_video_transcripts
from models.document import (
    CATEGORY_VIDEO,
    CATEGORY_WORK,
    Document,
    SOURCE_VIDEO,
    SOURCE_WORK,
    SOURCE_WIKI,
    SOURCE_WEB,
)

logger = logging.getLogger(__name__)


# ==================== 预置著作库 ====================

def load_preset_works(character_name: str) -> list[Document]:
    """
    从 assets/works/<人物目录>/ 加载预置著作全文。
    优先使用注册表里的 works_dir 字段；未注册的人物直接按名字匹配目录。
    """
    import json
    from pathlib import Path

    from config import ASSETS_DIR

    registry = {}
    reg_path = ASSETS_DIR / "person_registry.json"
    if reg_path.exists():
        try:
            registry = json.loads(reg_path.read_text(encoding="utf-8"))
        except Exception:
            registry = {}

    # 注册表中别名映射到标准 key，找不到则直接用输入名
    key = character_name
    for reg_key, info in (registry or {}).items():
        if character_name == reg_key or character_name in info.get("aliases", []):
            key = reg_key
            break

    info = registry.get(key, {}) if registry else {}
    works_subdir = info.get("works_dir") or key
    works_dir = WORKS_DIR / works_subdir
    if not works_dir.exists():
        logger.debug("预置著作目录不存在: %s", works_dir)
        return []

    docs: list[Document] = []
    for txt_file in sorted(works_dir.glob("*.txt")):
        try:
            raw = txt_file.read_text(encoding="utf-8")
        except Exception as e:
            logger.warning("读取预置著作 %s 失败: %s", txt_file, e)
            continue
        title = txt_file.stem
        era = info.get("era", "")
        file_docs = to_documents(
            character_name=character_name,
            source_type=SOURCE_WORK,
            source_detail=f"《{title}》（预置）",
            raw_text=raw,
            title=title,
            era=era,
            language="zh",
            category=CATEGORY_WORK,
            cleaner=None,
        )
        docs.extend(file_docs)

    logger.info("预置著作库为 %s 加载 %d 段", character_name, len(docs))
    return docs


# ==================== 段落化辅助 ====================

def _snippets_to_documents(
    character_name: str, snippets: list[RawSnippet]
) -> list[Document]:
    """把 RawSnippet 列表统一加工为 Document 段落。"""
    docs: list[Document] = []
    for snip in snippets:
        docs.extend(to_documents(
            character_name=character_name,
            source_type=snip.source_type,
            source_detail=snip.source_detail,
            raw_text=snip.content,
            title=snip.title,
            language=snip.language,
            category=getattr(snip, "category", "") or "",
            cleaner=clean_html if "<" in snip.content else None,
        ))
    return docs


def _global_dedup(docs: list[Document]) -> list[Document]:
    """全局去重（按规范化内容 SHA-1，首次出现保留）。

    不能用 set 过滤，否则重复 content 的两个 Document 都会被保留。
    """
    seen: set[str] = set()
    out: list[Document] = []
    for d in docs:
        norm = normalize_text(d.content)
        if not norm:
            continue
        h = hashlib.sha1(norm.encode("utf-8")).hexdigest()
        if h in seen:
            continue
        seen.add(h)
        out.append(d)
    return out


# ==================== 统一采集入口（按四步链路） ====================

def collect_all_sources(
    character_name: str,
    *,
    identity: PersonIdentity | None = None,
    enable_preset: bool = True,
    enable_works: bool = True,
    enable_biographies: bool = True,
    enable_history: bool = True,
    enable_video: bool = True,
    # 旧参数兼容
    enable_wiki: bool = True,   # noqa: ARG001（wiki 已纳入各 step 内）
    enable_web: bool = True,    # noqa: ARG001（web 已纳入各 step 内）
) -> list[Document]:
    """
    按四步链路采集所有数据源，返回去重后的 Document 列表。

    每步独立 try-except，任一步失败不影响其他步。

    参数：
      identity: 已识别的人物身份（必须含 work/biography/history_queries）；
                None 时退化为仅用名字做基础采集（兼容旧调用）。
      enable_preset: 是否加载预置著作库
      enable_works: 是否执行 Step 2（本人著作）
      enable_biographies: 是否执行 Step 3（他人传记）
      enable_history: 是否执行 Step 4（权威史料）
      enable_video: 是否采集视频字幕
    """
    canonical_name = identity.name if identity and identity.name else character_name
    all_docs: list[Document] = []
    source_counts: dict[str, int] = {}

    def _add(source: str, new_docs: list[Document]) -> None:
        all_docs.extend(new_docs)
        source_counts[source] = source_counts.get(source, 0) + len(new_docs)

    # Step 0：预置著作库
    if enable_preset:
        try:
            _add(SOURCE_WORK + "(preset)", load_preset_works(canonical_name))
        except Exception as e:
            logger.warning("预置著作加载失败: %s", e)

    if identity is None:
        # 兼容旧调用：无 identity 时只做基础采集
        from core.search_engine import collect_base_sources
        try:
            snippets = collect_base_sources(canonical_name)
            _add(SOURCE_WEB, _snippets_to_documents(canonical_name, snippets))
        except Exception as e:
            logger.warning("基础资料采集失败: %s", e)
    else:
        # Step 2：本人著作/自传
        if enable_works:
            try:
                snippets = search_person_works(identity)
                _add(SOURCE_WEB + "/wiki(work)", _snippets_to_documents(canonical_name, snippets))
            except Exception as e:
                logger.warning("Step2 本人著作采集失败: %s", e)

        # Step 3：他人传记/评传
        if enable_biographies:
            try:
                snippets = search_person_biographies(identity)
                _add(SOURCE_WEB + "(biography)", _snippets_to_documents(canonical_name, snippets))
            except Exception as e:
                logger.warning("Step3 他人传记采集失败: %s", e)

        # Step 4：权威史料/时代背景
        if enable_history:
            try:
                snippets = search_person_history(identity)
                _add(SOURCE_WEB + "(history)", _snippets_to_documents(canonical_name, snippets))
            except Exception as e:
                logger.warning("Step4 权威史料采集失败: %s", e)

    # Step 5：可选视频字幕
    if enable_video:
        try:
            _add(SOURCE_VIDEO, collect_video_transcripts(canonical_name))
        except Exception as e:
            logger.warning("视频字幕采集失败: %s", e)

    # 全局去重（按规范化内容 SHA-1，首次出现保留）
    before = len(all_docs)
    all_docs = _global_dedup(all_docs)
    if before != len(all_docs):
        logger.info("去重：%d → %d 段", before, len(all_docs))

    logger.info(
        "为 %s 采集完成，共 %d 段。来源分布: %s",
        canonical_name, len(all_docs), source_counts,
    )
    return all_docs


def summarize_sources(docs: list[Document]) -> dict[str, int]:
    """统计各来源类型段数，用于人物档案展示。"""
    counts: dict[str, int] = {}
    for d in docs:
        # 归一到主来源类型（著作/百科/网页/视频字幕）
        label = d.source_label()
        counts[label] = counts.get(label, 0) + 1
    return counts


# ==================== 数据源健康检查 ====================

def health_check() -> list[tuple[str, bool, str]]:
    """
    逐个探测数据源可用性，返回 [(名称, 是否可用, 说明), ...]。

    用途：`python launcher.py check` 或构建档案前的自检。
    背景：采集链路一旦全线失败，前端只会显示「共 0 段资料」，
    看不出到底是代理没开、站点被墙，还是代码不兼容，因此需要显式探测。
    """
    from core.search_engine import web_search_health, wikipedia_health
    from core.web_fetcher import fetch_url

    results: list[tuple[str, bool, str]] = []

    # 1. 预置著作库（离线，永远可用）
    preset_ok = WORKS_DIR.exists()
    results.append((
        "预置著作库",
        preset_ok,
        f"目录 {WORKS_DIR}" if preset_ok else f"目录不存在：{WORKS_DIR}",
    ))

    # 2. 网页正文抓取（顺带覆盖代理与 TLS 是否可用）
    #    注意：探针不能选 bing.com 这类「首页几乎全是导航/表单、没有正文段落」的页面，
    #    它们会被正文长度过滤规则判为无效页，从而误报失败。这里选正文稳定的页面。
    probe_urls = (
        "https://example.com",
        "https://www.iana.org/help/example-domains",
    )
    last_reason = ""
    fetch_ok = False
    fetch_detail = ""
    for url in probe_urls:
        try:
            page = fetch_url(url, max_chars=1000)
        except Exception as e:  # noqa: BLE001
            last_reason = f"抛出异常：{type(e).__name__}: {e}"
            continue
        if page is not None and len(page.text) >= 200:
            fetch_ok = True
            fetch_detail = f"成功抓取 {len(page.text)} 字（{url}）"
            break
        last_reason = f"{url} 抓取失败或正文过短"

    if fetch_ok:
        results.append(("网页抓取", True, fetch_detail))
    else:
        from core.web_fetcher import proxy_hint

        results.append(("网页抓取", False, f"{last_reason}。{proxy_hint()}"))

    # 3. 网页检索（DuckDuckGo → Bing RSS 兜底）
    ok, detail = web_search_health()
    results.append(("网页检索", ok, detail))

    # 4. 维基百科（wikipediaapi → REST 兜底）
    ok, detail = wikipedia_health()
    results.append(("维基百科", ok, detail))

    return results


def format_health_report(results: list[tuple[str, bool, str]]) -> str:
    """把健康检查结果格式化成可读报告。"""
    lines = ["数据源自检："]
    for name, ok, detail in results:
        lines.append(f"  {'✅' if ok else '❌'} {name}：{detail}")
    failed = [name for name, ok, _ in results if not ok]
    lines.append("")
    if failed:
        lines.append(
            f"⚠️ 有 {len(failed)} 项数据源不可用：{'、'.join(failed)}。"
            "档案仍可构建，但会退化为「仅用 LLM 内置知识」，质量较低。"
        )
        lines.append(
            "   最常见原因：未配置代理或代理未启动。"
            "请在 .env 中设置 HTTPS_PROXY=http://127.0.0.1:7890（端口按你的代理软件填），"
            "或确认代理软件已运行。"
        )
    else:
        lines.append("✅ 所有数据源均可用。")
    return "\n".join(lines)


def _material_category(doc: Document) -> str:
    """兼容旧数据：无 category 字段时，从来源详情推断资料类别。"""
    detail = doc.source_detail or ""
    if doc.source_type == SOURCE_WORK or "著作" in detail:
        return "本人著作/自传"
    if "传记" in detail:
        return "他人传记/评传"
    if "史料" in detail:
        return "权威史料/时代背景"
    if doc.source_type == SOURCE_WIKI:
        return "百科资料"
    if doc.source_type == SOURCE_VIDEO:
        return "视频字幕"
    return "其他网页资料"


def format_collected_materials(docs: list[Document], max_items_per_category: int = 12) -> str:
    """生成可直接展示在前端的具体采集清单（按资料而非段落去重）。"""
    if not docs:
        return "### 📚 本次采用的具体资料\n\n未采集到外部资料。"

    grouped: dict[str, dict[tuple[str, str], int]] = {}
    for doc in docs:
        category = doc.category_label() if doc.category else _material_category(doc)
        title = (doc.title or "").strip()
        detail = (doc.source_detail or doc.source_label()).strip()
        # 标题通常描述具体页面/作品，detail 描述检索路线或来源平台。
        key = (title or detail or "未命名资料", detail)
        items = grouped.setdefault(category, {})
        items[key] = items.get(key, 0) + 1

    preferred_order = [
        "本人著作/自传",
        "他人传记/评传",
        "权威史料/时代背景",
        "百科资料",
        "视频字幕",
        "其他网页资料",
    ]
    lines = ["### 📚 本次采用的具体资料"]
    for category in preferred_order:
        items = grouped.get(category)
        if not items:
            continue
        total_segments = sum(items.values())
        lines.append(f"\n**{category}**（{len(items)} 项，{total_segments} 段）")
        entries = list(items.items())
        for (title, detail), count in entries[:max_items_per_category]:
            source_note = f"；来源：{detail}" if detail and detail != title else ""
            lines.append(f"- {title}（{count} 段{source_note}）")
        remaining = len(entries) - max_items_per_category
        if remaining > 0:
            lines.append(f"- ……另有 {remaining} 项（为避免页面过长已折叠）")
    return "\n".join(lines)
