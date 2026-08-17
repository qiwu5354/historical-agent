"""
人物档案构建器（核心模块）。

职责：
  1. 先用 LLM + 联网搜索识别用户输入的人物身份和学者类别；
  2. 再按识别结果采集本人著作、传记、历史背景等一手/权威资料；
  3. 把多源资料通过 LLM 提炼为结构化的 Character 档案。
"""
from __future__ import annotations

import logging
from typing import Iterator

from core.data_collector import collect_all_sources, summarize_sources
from core.llm_client import get_llm, LLMError
from core.person_identifier import PersonIdentity, identify_person
from models.character import Character
from models.document import Document, SOURCE_WORK, SOURCE_LABELS
from prompts.extraction_prompts import build_extraction_messages
from prompts.category_prompts import normalize_categories

logger = logging.getLogger(__name__)


# 喂给 LLM 做档案提取的最大字符数（控制 token 成本）
MAX_MATERIAL_CHARS = 12000
# 各来源类型用于提取的配额（优先著作）
SOURCE_QUOTA = {
    SOURCE_WORK: 5000,      # 著作权重最高
    "wiki": 3000,
    "web": 2000,
    "video_subtitle": 2000,
}


# ==================== 选材：从所有文档中选出代表性段落 ====================

def select_representative_materials(docs: list[Document]) -> str:
    """
    从采集到的所有文档中，按来源类型配额选出代表性段落，拼接成给 LLM 的材料。
    优先选著作原文，其次百科、网页、字幕。
    """
    # 按来源分组
    by_source: dict[str, list[Document]] = {}
    for d in docs:
        by_source.setdefault(d.source_type, []).append(d)

    materials: list[str] = []
    for source_type in (SOURCE_WORK, "wiki", "web", "video_subtitle"):
        group = by_source.get(source_type, [])
        if not group:
            continue
        quota = SOURCE_QUOTA.get(source_type, 1500)
        label = SOURCE_LABELS.get(source_type, source_type)
        buf = f"\n\n=====【来源：{label}】=====\n"
        used = 0
        for doc in group:
            if used >= quota:
                break
            piece = doc.content[:quota - used]
            buf += f"\n[{doc.source_detail}] {piece}\n"
            used += len(piece)
        materials.append(buf)

    full = "".join(materials)
    # 总量控制
    if len(full) > MAX_MATERIAL_CHARS:
        full = full[:MAX_MATERIAL_CHARS] + "\n\n[...资料过长，已截断...]"
    return full


# ==================== 从 JSON 构建 Character ====================

def _character_from_json(
    data: dict,
    character_name: str,
    docs: list[Document],
    categories: list[str] | None = None,
) -> Character:
    """把 LLM 输出的 JSON dict 转为 Character 对象，并补充统计字段。"""
    sources_summary = summarize_sources(docs)
    # 优先使用识别阶段确定的类别；如果提取结果里有，也做归一化合并。
    extracted_categories = normalize_categories(data.get("categories") or [])
    merged_categories: list[str] = []
    for cat in list(categories or []) + extracted_categories:
        if cat not in merged_categories:
            merged_categories.append(cat)
    char = Character(
        name=data.get("name", character_name) or character_name,
        aliases=data.get("aliases", []) or [],
        categories=merged_categories or ["其他"],
        era=data.get("era", "") or "",
        nation=data.get("nation", "") or "",
        biography=data.get("biography", "") or "",
        core_thought=data.get("core_thought", "") or "",
        key_concepts=data.get("key_concepts", []) or [],
        major_works=data.get("major_works", []) or [],
        famous_speeches=data.get("famous_speeches", []) or [],
        historical_context=data.get("historical_context", "") or "",
        stance_guideline=data.get("stance_guideline", "") or "",
        doc_count=len(docs),
        sources_summary=sources_summary,
    )
    return char


# ==================== 对外入口 ====================

def build_character(
    character_name: str,
    *,
    identity: PersonIdentity | None = None,
    docs: list[Document] | None = None,
    enable_video: bool = True,
) -> tuple[Character, list[Document]]:
    """
    构建人物档案。

    参数：
      character_name: 用户输入的人物名
      identity: 可选，已识别出的人物身份；None 则内部调用 identify_person
      docs: 可选，外部已采集好的文档（避免重复采集）；None 则内部采集
      enable_video: 是否启用视频字幕采集（耗时长，可选关闭）

    返回：(Character 档案, 所有 Document 列表)
    """
    character_name = character_name.strip()
    logger.info("开始为 %s 构建档案", character_name)

    # 0. 识别标准人物身份
    if identity is None:
        identity = identify_person(character_name)
    canonical_name = identity.name or character_name

    # 1. 采集数据（若未外部提供）
    if docs is None:
        docs = collect_all_sources(
            canonical_name,
            identity=identity,
            enable_video=enable_video,
        )

    if not docs:
        logger.warning("未采集到 %s 的任何资料，将用 LLM 内置知识构建（质量较低）", canonical_name)
        # 兜底：至少让 LLM 用自身知识构建，但不推荐
        materials = f"（系统未采集到关于「{canonical_name}」的外部资料，请基于你的历史知识提取档案，并标注可能不准确）"
    else:
        materials = select_representative_materials(docs)

    # 2. LLM 提取结构化档案
    messages = build_extraction_messages(canonical_name, materials)
    llm = get_llm()
    try:
        data = llm.chat_json(messages)
    except LLMError as e:
        logger.error("档案提取失败: %s", e)
        raise

    # 3. 组装 Character
    char = _character_from_json(data, canonical_name, docs, categories=identity.categories)
    # 保证人物主键统一使用识别出的标准名，避免别名导致数据错乱
    char.name = canonical_name
    logger.info("档案构建完成：%s（%d段资料，类别=%s，来源：%s）",
                char.name, char.doc_count, char.categories, char.sources_summary)
    return char, docs


def build_character_streaming(
    character_name: str,
    *,
    enable_video: bool = True,
) -> Iterator[tuple[str, float]]:
    """
    流式构建档案，用于 UI 实时展示进度。

    yield (进度文本, 进度百分比 0~100)；
    生成器结束时 return (Character, docs)，调用方通过 StopIteration.value 取得返回值。

    新流程：
      1. 联网 + LLM 识别人物身份和学者类别
      2. 按识别结果搜索本人著作、传记、历史背景
      3. 用 LLM 提炼结构化人物档案
    """
    character_name = character_name.strip()
    yield f"🔍 正在识别人物：「{character_name}」...\n", 5.0

    # 1. 人物识别
    try:
        identity = identify_person(character_name)
    except Exception as e:
        logger.exception("人物识别失败")
        raise RuntimeError(f"人物识别失败：{e}") from e

    yield (
        f"✅ 已识别：{identity.name}\n"
        f"   - 类别：{'、'.join(identity.categories) or '未分类'}\n"
        f"   - 简介：{identity.summary or '（无）'}\n\n",
        25.0,
    )

    # 2. 深度资料采集
    yield "🔎 正在搜索本人著作、传记、历史背景资料...\n", 30.0
    if enable_video:
        yield "  - 视频字幕（如启用且可用）...\n", 32.0

    docs = collect_all_sources(
        identity.name,
        identity=identity,
        enable_video=enable_video,
    )
    yield f"\n✅ 采集完成：共 {len(docs)} 段资料\n", 65.0
    if docs:
        src = summarize_sources(docs)
        src_str = "、".join(f"{k}:{v}段" for k, v in src.items())
        yield f"📊 来源分布：{src_str}\n\n", 68.0

    # 3. 提取档案
    yield "🧠 正在用 LLM 提炼核心思想、构建人物档案...\n", 70.0
    char, _ = build_character(
        identity.name,
        identity=identity,
        docs=docs,
        enable_video=False,
    )
    yield "\n✅ 档案构建完成！\n\n", 95.0
    yield char.summary(), 100.0

    # 通过生成器返回值把档案和文档交给调用方（StopIteration.value）
    return char, docs
