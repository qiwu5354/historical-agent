"""
统一数据采集器：聚合所有数据源，返回统一的 Document 列表。

新数据流：
  1. 人物识别阶段已经得到标准身份和学者类别；
  2. 按 LLM 生成的检索词，分别搜索本人著作、他人传记、历史阶段权威史料；
  3. 补充预置著作、维基百科、可选视频字幕等；
  4. 所有源的原始文本都经过 text_processor 加工为 Document 段落。
"""
from __future__ import annotations

import logging

from config import WORKS_DIR
from core.person_identifier import PersonIdentity
from core.search_engine import collect_base_sources, collect_scholar_sources
from core.text_processor import to_documents, dedup, clean_html
from core.video_transcript import collect_video_transcripts
from models.document import Document, SOURCE_WORK, SOURCE_WIKI, SOURCE_WEB, SOURCE_VIDEO

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
            cleaner=None,
        )
        docs.extend(file_docs)

    logger.info("预置著作库为 %s 加载 %d 段", character_name, len(docs))
    return docs


# ==================== 统一采集入口 ====================

def collect_all_sources(
    character_name: str,
    *,
    identity: PersonIdentity | None = None,
    enable_preset: bool = True,
    enable_wiki: bool = True,
    enable_web: bool = True,
    enable_video: bool = True,
) -> list[Document]:
    """
    采集所有数据源，返回去重后的 Document 列表。

    新流程：
      1. 预置著作库（离线高质量文本）
      2. 根据 LLM 识别出的检索词，搜索本人著作、传记、历史背景资料
      3. 补充多语言维基
      4. 可选视频字幕

    每个源独立 try-except，任一源失败不影响其他源。
    """
    canonical_name = identity.name if identity and identity.name else character_name
    all_docs: list[Document] = []
    source_counts: dict[str, int] = {}

    def _add(source: str, new_docs: list[Document]) -> None:
        all_docs.extend(new_docs)
        source_counts[source] = source_counts.get(source, 0) + len(new_docs)

    # 1. 预置著作库
    if enable_preset:
        try:
            _add(SOURCE_WORK + "(preset)", load_preset_works(canonical_name))
        except Exception as e:
            logger.warning("预置著作加载失败: %s", e)

    # 2. 深度资料检索：本人著作 / 传记 / 历史背景 / 维基
    if enable_wiki or enable_web:
        try:
            if identity is not None:
                snippets = collect_scholar_sources(
                    identity,
                    enable_web=enable_web,
                    enable_wiki=enable_wiki,
                )
            else:
                snippets = collect_base_sources(
                    canonical_name, enable_web=enable_web, enable_wiki=enable_wiki
                )
            for snip in snippets:
                docs = to_documents(
                    character_name=canonical_name,
                    source_type=snip.source_type,
                    source_detail=snip.source_detail,
                    raw_text=snip.content,
                    title=snip.title,
                    language=snip.language,
                    cleaner=clean_html if "<" in snip.content else None,
                )
                _add(snip.source_type, docs)
        except Exception as e:
            logger.warning("深度资料采集失败: %s", e)

    # 3. 视频字幕
    if enable_video:
        try:
            _add(SOURCE_VIDEO, collect_video_transcripts(canonical_name))
        except Exception as e:
            logger.warning("视频字幕采集失败: %s", e)

    # 全局去重（按内容）
    before = len(all_docs)
    seen_contents = dedup([d.content for d in all_docs])
    # 保持 Document 关联
    keep = set(seen_contents)
    all_docs = [d for d in all_docs if d.content in keep]
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
