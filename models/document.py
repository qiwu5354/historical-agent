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

# 资料类别（面向用户的资料分类，独立于技术来源类型）
CATEGORY_WORK = "work"                   # 本人著作/自传/文章
CATEGORY_BIOGRAPHY = "biography"         # 他人撰写的传记/评传/研究
CATEGORY_HISTORY = "history"             # 权威史料/同时代历史背景
CATEGORY_WIKI = "wiki"                   # 百科资料
CATEGORY_VIDEO = "video"                 # 视频字幕
CATEGORY_WEB = "web"                     # 其他网页资料

CATEGORY_LABELS = {
    CATEGORY_WORK: "本人著作/自传",
    CATEGORY_BIOGRAPHY: "他人传记/评传",
    CATEGORY_HISTORY: "权威史料/时代背景",
    CATEGORY_WIKI: "百科资料",
    CATEGORY_VIDEO: "视频字幕",
    CATEGORY_WEB: "其他网页资料",
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
    category: str = ""                    # 资料类别，见 CATEGORY_* 常量（空=未标注）
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

    def category_label(self) -> str:
        """资料类别中文标签；未标注时回退到来源标签"""
        return CATEGORY_LABELS.get(self.category) or self.source_label()
