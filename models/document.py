"""
知识库文档数据模型。

Document 是 RAG 检索的最小单元（一个段落），
source_type 标记其来源（著作/维基/网页/视频字幕），便于溯源展示。
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any


# 文档来源类型常量
SOURCE_WORK = "work"                     # 人物著作全文（预置或爬取）
SOURCE_WIKI = "wiki"                     # 维基百科
SOURCE_WEB = "web"                       # DuckDuckGo 网页摘要
SOURCE_VIDEO = "video_subtitle"          # 视频（YouTube/Bilibili）字幕

ALL_SOURCE_TYPES = (SOURCE_WORK, SOURCE_WIKI, SOURCE_WEB, SOURCE_VIDEO)

SOURCE_LABELS = {
    SOURCE_WORK: "著作",
    SOURCE_WIKI: "百科",
    SOURCE_WEB: "网页",
    SOURCE_VIDEO: "视频字幕",
}


@dataclass
class Document:
    """知识库文档段落（RAG 最小检索单元）"""
    doc_id: str                           # 全局唯一 ID
    character_name: str                   # 所属人物
    source_type: str                      # 来源类型，见上方常量
    source_detail: str = ""               # 细节："《梦的解析》" | "YouTube:xxx" | "维基百科"
    title: str = ""                       # 标题/章节
    era: str = ""                         # 写作/发布年代
    language: str = "zh"                  # 原文语种
    content: str = ""                     # 段落文本
    # 向量不在此存储（FAISS 独立管理），doc_id 作为关联键

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Document:
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in known})

    def source_label(self) -> str:
        """来源中文标签，用于 UI 展示"""
        return SOURCE_LABELS.get(self.source_type, self.source_type)
