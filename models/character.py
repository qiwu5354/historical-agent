"""
人物档案数据模型。

由 character_builder 从多源资料中用 LLM 提取生成。
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any


@dataclass
class Character:
    """历史人物/学者档案"""
    name: str                               # 标准名（如"弗洛伊德"）
    aliases: list[str] = field(default_factory=list)        # 别名
    categories: list[str] = field(default_factory=list)     # 学者类别：哲学家/经济学家/心理学家...
    era: str = ""                           # 生卒年（如"1818-1883"）
    nation: str = ""                        # 国家/民族
    biography: str = ""                     # 生平概述
    core_thought: str = ""                  # 核心思想（详细）
    key_concepts: list[str] = field(default_factory=list)   # 关键概念
    major_works: list[str] = field(default_factory=list)    # 代表著作
    famous_speeches: list[str] = field(default_factory=list)  # 经典演讲/文章
    historical_context: str = ""            # 思想产生的时代背景
    stance_guideline: str = ""              # 立场指南（角色扮演用）
    doc_count: int = 0                      # 知识库文档数（构建后填充）
    sources_summary: dict[str, int] = field(default_factory=dict)  # 数据来源统计
    created_at: str = field(
        default_factory=lambda: datetime.now().isoformat(timespec="seconds")
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Character:
        # 只取已知字段，忽略多余键，保证向前兼容
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in known})

    def summary(self) -> str:
        """生成给人看的简短摘要（用于 UI 展示）"""
        parts = [f"## {self.name}"]
        if self.era:
            parts.append(f"**年代**：{self.era}")
        if self.nation:
            parts.append(f"**国家**：{self.nation}")
        if self.aliases:
            parts.append(f"**别名**：{'、'.join(self.aliases)}")
        if self.categories:
            parts.append(f"**类别**：{'、'.join(self.categories)}")
        if self.core_thought:
            parts.append(f"\n### 核心思想\n{self.core_thought}")
        if self.key_concepts:
            parts.append(f"\n### 关键概念\n{ '、'.join(self.key_concepts) }")
        if self.major_works:
            parts.append(f"\n### 代表著作\n{ '、'.join(self.major_works) }")
        if self.sources_summary:
            src = "、".join(f"{k}:{v}段" for k, v in self.sources_summary.items())
            parts.append(f"\n### 知识库来源\n{src}（共{self.doc_count}段）")
        return "\n".join(parts)
