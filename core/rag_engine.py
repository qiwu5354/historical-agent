"""
RAG 引擎：双后端语义检索。

后端策略（自动选择）：
  1. sentence-transformers + FAISS  —— 首选，语义检索质量最高
  2. TF-IDF + 余弦相似度（scikit-learn） —— 回退，无需 torch，
     当 torch/sentence-transformers 加载失败时自动启用

每个角色对应一个独立的索引，按人物名字组织：
  data/faiss_index/<character_name>/
检索后通过 doc_id 与 SQLite documents 表关联。
"""
from __future__ import annotations

import hashlib
import logging
import pickle
from pathlib import Path
from typing import Any

import numpy as np

from config import FAISS_DIR, settings
from models.document import Document

logger = logging.getLogger(__name__)


def _stable_doc_id(doc_id: str) -> int:
    """
    把 doc_id 映射为稳定的 int64（FAISS 内部 ID）。
    不能用内置 hash()：Python 对字符串的 hash 每次进程启动都会随机化，
    会导致跨进程检索时 ID 对应不上。改用 md5 保证跨进程稳定。
    """
    return int(hashlib.md5(doc_id.encode("utf-8")).hexdigest(), 16) & 0x7FFFFFFFFFFFFFFF


# 后端类型常量
BACKEND_TRANSFORMER = "transformer"   # sentence-transformers + FAISS
BACKEND_TFIDF = "tfidf"               # scikit-learn TF-IDF
_active_backend: str | None = None    # 运行时确定，None=未探测

# 全局共享的 embedding 模型（加载耗时，单例）
_embedding_model: Any = None


# ==================== 后端探测 ====================

def _try_load_transformer():
    """尝试加载 sentence-transformers 模型，成功则返回模型，失败返回 None。"""
    global _embedding_model
    if _embedding_model is not None:
        return _embedding_model
    try:
        from sentence_transformers import SentenceTransformer
        logger.info("加载向量模型: %s ...", settings.rag.embedding_model)
        _embedding_model = SentenceTransformer(
            settings.rag.embedding_model, device="cpu"
        )
        # 做一次空编码验证可用
        _embedding_model.encode(["测试"], normalize_embeddings=True)
        logger.info("向量模型加载完成（transformer 后端）")
        return _embedding_model
    except Exception as e:
        logger.warning(
            "sentence-transformers 不可用（%s），将回退到 TF-IDF 后端。"
            "如需更高质量的语义检索，请修复 torch 安装。", e
        )
        return None


def detect_backend() -> str:
    """
    探测可用后端（只探测一次，结果缓存）。
    transformer 可用则用它，否则回退到 tfidf。
    """
    global _active_backend
    if _active_backend is not None:
        return _active_backend
    if _try_load_transformer() is not None:
        _active_backend = BACKEND_TRANSFORMER
    else:
        _active_backend = BACKEND_TFIDF
        logger.info("已启用 TF-IDF 后端（无需 torch，开箱即用）")
    return _active_backend


# ==================== 文本编码 ====================

def _encode_transformer(texts: list[str]) -> np.ndarray:
    """用 sentence-transformers 编码（L2 归一化）。"""
    model = _try_load_transformer()
    vecs = model.encode(
        texts, normalize_embeddings=True, show_progress_bar=False, convert_to_numpy=True
    )
    return vecs.astype(np.float32)


# ==================== 索引路径 ====================

def _index_dir(character_name: str) -> Path:
    safe = character_name.replace("/", "_").replace("\\", "_")
    d = FAISS_DIR / safe
    d.mkdir(parents=True, exist_ok=True)
    return d


def _index_path(character_name: str) -> Path:
    return _index_dir(character_name) / "index.faiss"


def _tfidf_path(character_name: str) -> Path:
    return _index_dir(character_name) / "tfidf.pkl"


# ==================== 构建索引 ====================

def build_index(character_name: str, docs: list[Document]) -> int:
    """
    为某人物构建索引（自动选择后端）。
    返回索引的文档数。
    """
    if not docs:
        logger.warning("无可索引的文档: %s", character_name)
        return 0

    backend = detect_backend()
    if backend == BACKEND_TRANSFORMER:
        return _build_index_faiss(character_name, docs)
    else:
        return _build_index_tfidf(character_name, docs)


def _build_index_faiss(character_name: str, docs: list[Document]) -> int:
    """transformer 后端：构建 FAISS 索引。"""
    import faiss

    texts = [d.content for d in docs]
    doc_ids = [d.doc_id for d in docs]
    vectors = _encode_transformer(texts)
    dim = vectors.shape[1]

    index = faiss.IndexFlatIP(dim)
    index = faiss.IndexIDMap(index)
    id_array = np.array(
        [_stable_doc_id(did) for did in doc_ids], dtype=np.int64
    )
    index.add_with_ids(vectors, id_array)

    faiss.write_index(index, str(_index_path(character_name)))
    id_map_path = _index_dir(character_name) / "doc_ids.npy"
    np.save(id_map_path, np.array(doc_ids, dtype=object), allow_pickle=True)

    logger.info("[FAISS] 为 %s 构建索引完成：%d 个向量", character_name, index.ntotal)
    return index.ntotal


def _build_index_tfidf(character_name: str, docs: list[Document]) -> int:
    """TF-IDF 后端：拟合并向量化所有文档。"""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity  # noqa: F401

    texts = [d.content for d in docs]
    doc_ids = [d.doc_id for d in docs]

    # 中文用 char-level n-gram（无需分词即可工作）
    vectorizer = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(1, 2), min_df=1
    )
    try:
        matrix = vectorizer.fit_transform(texts)  # sparse CSR
    except ValueError as e:
        logger.warning("TF-IDF 拟合失败（可能语料过短）: %s", e)
        return 0

    payload = {
        "vectorizer": vectorizer,
        "matrix": matrix,
        "doc_ids": doc_ids,
    }
    with open(_tfidf_path(character_name), "wb") as f:
        pickle.dump(payload, f)

    logger.info("[TF-IDF] 为 %s 构建索引完成：%d 个文档", character_name, matrix.shape[0])
    return matrix.shape[0]


# ==================== 检索 ====================

def search(
    character_name: str, query: str, top_k: int | None = None
) -> list[tuple[Document, float]]:
    """
    在某人物的索引中检索，返回 [(Document, 相似度分数), ...]。
    需要 SQLite 中已有对应文档。
    """
    from db.repository import get_documents

    top_k = top_k or settings.rag.top_k

    backend = detect_backend()
    if backend == BACKEND_TRANSFORMER and _index_path(character_name).exists():
        results = _search_faiss(character_name, query, top_k)
    elif _tfidf_path(character_name).exists():
        results = _search_tfidf(character_name, query, top_k)
    else:
        logger.warning("人物 %s 的索引不存在，跳过 RAG", character_name)
        return []

    # 取回文档
    all_docs = {d.doc_id: d for d in get_documents(character_name)}
    return [
        (all_docs[did], score)
        for did, score in results
        if did in all_docs
    ]


def _search_faiss(
    character_name: str, query: str, top_k: int
) -> list[tuple[str, float]]:
    """FAISS 后端检索，返回 [(doc_id, score), ...]"""
    import faiss

    index = faiss.read_index(str(_index_path(character_name)))
    if index.ntotal == 0:
        return []

    q_vec = _encode_transformer([query])
    top_k = min(top_k, index.ntotal)
    scores, ids = index.search(q_vec, top_k)

    id_map_path = _index_dir(character_name) / "doc_ids.npy"
    doc_ids: list[str] = np.load(id_map_path, allow_pickle=True).tolist()
    hash_to_id = {_stable_doc_id(d): d for d in doc_ids}

    out: list[tuple[str, float]] = []
    for score, hid in zip(scores[0], ids[0]):
        if hid == -1:
            continue
        doc_id = hash_to_id.get(int(hid))
        if doc_id:
            out.append((doc_id, float(score)))
    return out


def _search_tfidf(
    character_name: str, query: str, top_k: int
) -> list[tuple[str, float]]:
    """TF-IDF 后端检索，返回 [(doc_id, score), ...]"""
    from sklearn.metrics.pairwise import cosine_similarity

    with open(_tfidf_path(character_name), "rb") as f:
        payload = pickle.load(f)
    vectorizer = payload["vectorizer"]
    matrix = payload["matrix"]
    doc_ids: list[str] = payload["doc_ids"]

    if matrix.shape[0] == 0:
        return []

    q_vec = vectorizer.transform([query])
    sims = cosine_similarity(q_vec, matrix)[0]  # 长度 = 文档数
    top_k = min(top_k, len(sims))
    # 取 top_k 的下标（按相似度降序）
    top_idx = np.argsort(sims)[::-1][:top_k]
    out: list[tuple[str, float]] = []
    for i in top_idx:
        score = float(sims[i])
        if score <= 0:   # 完全不相关则不再补
            continue
        out.append((doc_ids[int(i)], score))
    return out


# ==================== 构建检索上下文 ====================

def build_context(
    character_name: str, query: str, top_k: int | None = None
) -> str:
    """
    检索并拼接成给 LLM 的"参考资料"文本块。
    若检索失败或无索引，返回空字符串（对话仍可继续，仅缺少 RAG 增强）。
    """
    try:
        results = search(character_name, query, top_k)
    except Exception as e:
        logger.warning("RAG 检索失败，将以无增强方式继续: %s", e)
        return ""

    if not results:
        return ""

    blocks: list[str] = []
    for doc, score in results:
        blocks.append(
            f"【{doc.source_label()}｜{doc.source_detail}】(相关度 {score:.2f})\n{doc.content}"
        )
    return "\n\n---\n\n".join(blocks)


def current_backend() -> str:
    """返回当前正在使用的后端名称（供 UI 展示）。"""
    return detect_backend()
