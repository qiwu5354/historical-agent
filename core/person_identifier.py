"""
人物识别模块。

职责：根据用户输入的名字，先做一轮轻量联网检索，再利用 LLM 判断该名字对应的
标准人物身份、学者类别，以及后续深度检索所需的搜索词。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from core.llm_client import get_llm
from core.search_engine import collect_identity_snippets, format_raw_snippets
from prompts.identification_prompt import build_identification_messages
from prompts.category_prompts import normalize_categories

logger = logging.getLogger(__name__)


@dataclass
class PersonIdentity:
    """识别出的人物身份信息，用于后续采集与对话。"""
    name: str
    aliases: list[str] = field(default_factory=list)
    categories: list[str] = field(default_factory=list)
    era: str = ""
    nation: str = ""
    summary: str = ""
    works: list[str] = field(default_factory=list)
    work_queries: list[str] = field(default_factory=list)
    biography_queries: list[str] = field(default_factory=list)
    history_queries: list[str] = field(default_factory=list)
    language: str = "zh"

    @classmethod
    def from_dict(
        cls, data: dict[str, Any], fallback_name: str = ""
    ) -> "PersonIdentity":
        if not isinstance(data, dict):
            data = {}
        name = str(data.get("name") or fallback_name or "").strip()
        if not name:
            name = fallback_name.strip()
        categories = normalize_categories(data.get("categories") or [])

        def _str_list(value: Any) -> list[str]:
            if isinstance(value, str):
                return [value]
            return [str(v).strip() for v in (value or []) if str(v).strip()]

        return cls(
            name=name,
            aliases=_str_list(data.get("aliases")),
            categories=categories,
            era=str(data.get("era") or "").strip(),
            nation=str(data.get("nation") or "").strip(),
            summary=str(data.get("summary") or "").strip(),
            works=_str_list(data.get("works")),
            work_queries=_str_list(data.get("work_queries")),
            biography_queries=_str_list(data.get("biography_queries")),
            history_queries=_str_list(data.get("history_queries")),
            language=str(data.get("language") or "zh").strip() or "zh",
        )

    @property
    def all_queries(self) -> list[str]:
        """全部检索词，按作品、传记、历史背景排序。"""
        return self.work_queries + self.biography_queries + self.history_queries

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "aliases": self.aliases,
            "categories": self.categories,
            "era": self.era,
            "nation": self.nation,
            "summary": self.summary,
            "works": self.works,
            "work_queries": self.work_queries,
            "biography_queries": self.biography_queries,
            "history_queries": self.history_queries,
            "language": self.language,
        }


def _fallback_identity(input_name: str) -> PersonIdentity:
    """LLM 识别失败时的兜底身份。"""
    return PersonIdentity(
        name=input_name.strip(),
        aliases=[],
        categories=["其他"],
        summary=f"未识别到「{input_name}」的详细身份信息，请基于通用学者方式对话。",
        work_queries=[f"{input_name} 著作 全文", f"{input_name} works full text"],
        biography_queries=[f"{input_name} 传记", f"{input_name} biography"],
        history_queries=[f"{input_name} 时代背景 历史", f"{input_name} historical background"],
    )


def identify_person(input_name: str) -> PersonIdentity:
    """
    识别用户输入的人物名。

    流程：
      1. 先用轻量搜索获取候选片段。
      2. 把片段交给 LLM，输出结构化身份信息。
      3. 如果 LLM 或搜索失败，返回基础身份，保证后续流程仍可继续。
    """
    input_name = input_name.strip()
    if not input_name:
        return _fallback_identity("未知人物")

    snippets: list = []
    try:
        snippets = collect_identity_snippets(input_name)
    except Exception as e:
        logger.warning("人物识别前置搜索失败，将仅依赖 LLM 知识: %s", e)

    snippets_text = format_raw_snippets(snippets)
    messages = build_identification_messages(input_name, snippets_text)
    try:
        data = get_llm().chat_json(messages)
    except Exception as e:
        logger.warning("LLM 人物识别失败，使用兜底身份: %s", e)
        data = {}

    identity = PersonIdentity.from_dict(data, fallback_name=input_name)
    if not identity.categories:
        identity.categories = ["其他"]
    logger.info(
        "人物识别结果: %s -> %s | 类别=%s",
        input_name, identity.name, identity.categories,
    )
    return identity
