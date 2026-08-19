"""
文本加工：清洗、切分、去重、多语言处理。

把各数据源采集到的 RawSnippet / 裸文本，加工成可入库的 Document 段落列表。
是所有采集器共用的下游处理层。
"""
from __future__ import annotations

import hashlib
import logging
import re
import uuid

from config import settings
from models.document import Document

logger = logging.getLogger(__name__)


# ==================== 清洗 ====================

def clean_html(raw: str) -> str:
    """去除 HTML 标签（维基/网页等来源常用）"""
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        # 无 bs4 时用正则兜底
        return re.sub(r"<[^>]+>", "", raw)
    soup = BeautifulSoup(raw, "lxml")
    return soup.get_text(separator="\n")


def clean_subtitle(raw: str) -> str:
    """清洗字幕文本：去时间戳、序号、多余空白。"""
    lines = raw.splitlines()
    out: list[str] = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        # 跳过纯数字行（字幕序号）
        if line.isdigit():
            continue
        # 跳过时间轴行  00:00:01,000 --> 00:00:03,000
        if "-->" in line or re.match(r"^\d{1,2}:\d{2}", line):
            continue
        # 去掉行内时间戳 [00:12] 等
        line = re.sub(r"\[\d{1,2}:\d{2}(:\d{2})?\]", "", line)
        out.append(line)
    return "\n".join(out)


def normalize_text(text: str) -> str:
    """通用文本规整：合并连续空白、去首尾。"""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


# ==================== 切分 ====================

def split_paragraphs(text: str, max_chars: int | None = None) -> list[str]:
    """
    将文本切分为段落。
    优先按自然段落（空行分隔）切；超长段落再按句号硬切。
    """
    max_chars = max_chars or settings.rag.chunk_size
    text = normalize_text(text)
    if not text:
        return []
    raw_paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]

    chunks: list[str] = []
    for para in raw_paragraphs:
        if len(para) <= max_chars:
            chunks.append(para)
        else:
            # 按句号/问号/感叹号硬切
            sentences = re.split(r"(?<=[。！？!?\.])\s*", para)
            buf = ""
            for s in sentences:
                if not s:
                    continue
                if len(buf) + len(s) <= max_chars:
                    buf += s
                else:
                    if buf:
                        chunks.append(buf)
                    buf = s
            if buf:
                chunks.append(buf)
    return [c for c in chunks if len(c) >= 10]  # 过滤过短噪声段


# ==================== 去重 ====================

def dedup(contents: list[str], threshold: float = 0.9) -> list[str]:
    """
    去重：规范化文本后按内容哈希去重（O(n)）。

    历史版本用 3-gram Jaccard 相似度做模糊去重，但那是 O(n²)
    的双重循环，语料一旦上千段就会卡死。当前改为"规范化 + SHA-1
    哈希"，对整段完全重复的内容（重复爬取、预置著作与在线库
    重复）足够，且量级线性，几万段也是毫秒级完成。
    """
    kept: list[str] = []
    seen: set[str] = set()
    for text in contents:
        norm = normalize_text(text)
        if not norm:
            continue
        h = hashlib.sha1(norm.encode("utf-8")).hexdigest()
        if h in seen:
            continue
        seen.add(h)
        kept.append(text)
    return kept


# ==================== 组装为 Document ====================

def to_documents(
    *,
    character_name: str,
    source_type: str,
    source_detail: str,
    raw_text: str,
    title: str = "",
    era: str = "",
    language: str = "zh",
    category: str = "",
    cleaner=None,
) -> list[Document]:
    """
    把一段原始文本清洗+切分后，包装成 Document 列表。
    cleaner: 可选的清洗函数（clean_html / clean_subtitle / None）
    category: 资料类别（CATEGORY_*），用于前端分类展示与 RAG 溯源
    """
    if cleaner is not None:
        raw_text = cleaner(raw_text)
    chunks = split_paragraphs(raw_text)
    docs: list[Document] = []
    for idx, chunk in enumerate(chunks):
        chunk_hash = uuid.uuid5(uuid.NAMESPACE_DNS, chunk).hex[:8]
        docs.append(
            Document(
                doc_id=f"{character_name}-{source_type}-{chunk_hash}-{idx}",
                character_name=character_name,
                source_type=source_type,
                source_detail=source_detail,
                title=title,
                era=era,
                language=language,
                content=chunk,
                category=category,
            )
        )
    return docs
