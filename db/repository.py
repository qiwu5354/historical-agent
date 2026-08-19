"""
数据访问层：人物档案、文档、对话历史的 CRUD。
"""
from __future__ import annotations

import json
import logging
from typing import Any

from db.database import get_conn
from models.character import Character
from models.document import Document

logger = logging.getLogger(__name__)


# ==================== 人物档案 ====================

def save_character(character: Character) -> None:
    """保存/更新人物档案（按 name 主键 upsert）。"""
    with get_conn() as conn:
        conn.execute(
            """
            INSERT INTO characters (name, profile_json, created_at)
            VALUES (?, ?, ?)
            ON CONFLICT(name) DO UPDATE SET
                profile_json = excluded.profile_json,
                created_at = excluded.created_at
            """,
            (character.name, json.dumps(character.to_dict(), ensure_ascii=False), character.created_at),
        )


def get_character(name: str) -> Character | None:
    """按名字读取人物档案。"""
    with get_conn() as conn:
        row = conn.execute(
            "SELECT profile_json FROM characters WHERE name = ?", (name,)
        ).fetchone()
    if not row:
        return None
    return Character.from_dict(json.loads(row["profile_json"]))


def find_character_by_any_name(name: str) -> Character | None:
    """先按标准名精确查找，再按别名模糊匹配。"""
    name = name.strip()
    char = get_character(name)
    if char:
        return char
    lowered = name.lower()
    for candidate in list_characters():
        item = get_character(candidate)
        if not item:
            continue
        if any(lowered == alias.strip().lower() for alias in item.aliases):
            return item
    return None


def list_characters() -> list[str]:
    """列出所有已建档的人物名。"""
    with get_conn() as conn:
        rows = conn.execute("SELECT name FROM characters ORDER BY created_at DESC").fetchall()
    return [r["name"] for r in rows]


# ==================== 文档段落 ====================

def save_documents(docs: list[Document]) -> int:
    """批量保存文档段落。返回成功插入的条数。"""
    if not docs:
        return 0
    rows = [
        (d.doc_id, d.character_name, d.source_type, d.source_detail,
         d.title, d.era, d.language, d.category, d.content)
        for d in docs
    ]
    with get_conn() as conn:
        # executemany 批量插入：单事务提交，比逐条 execute 快一个量级
        # （一次采集可达数万段，逐条插入会拖慢整个构建）。
        cur = conn.executemany(
            """
            INSERT OR IGNORE INTO documents
            (doc_id, character_name, source_type, source_detail, title, era, language, category, content)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        inserted = cur.rowcount if cur.rowcount is not None and cur.rowcount > 0 else 0
    logger.info("保存 %d 条文档（去重后插入 %d）", len(rows), inserted)
    return inserted


def save_character_with_documents(character: Character, docs: list[Document]) -> int:
    """
    在一个事务内保存人物档案并替换其全部文档。

    先 upsert 人物，再删除旧文档，最后插入新文档。避免重建后残留过期资料。
    返回写入的文档条数。
    """
    rows = [
        (d.doc_id, d.character_name, d.source_type, d.source_detail,
         d.title, d.era, d.language, d.category, d.content)
        for d in docs
    ]
    with get_conn() as conn:
        conn.execute(
            """
            INSERT INTO characters (name, profile_json, created_at)
            VALUES (?, ?, ?)
            ON CONFLICT(name) DO UPDATE SET
                profile_json = excluded.profile_json,
                created_at = excluded.created_at
            """,
            (character.name, json.dumps(character.to_dict(), ensure_ascii=False), character.created_at),
        )
        conn.execute(
            "DELETE FROM documents WHERE character_name = ?", (character.name,)
        )
        if rows:
            conn.executemany(
                """
                INSERT OR IGNORE INTO documents
                (doc_id, character_name, source_type, source_detail, title, era, language, category, content)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
    logger.info(
        "保存人物 %s 及 %d 条文档（旧文档已替换）",
        character.name, len(rows),
    )
    return len(rows)



def get_documents(character_name: str) -> list[Document]:
    """读取某人物的所有文档段落。"""
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM documents WHERE character_name = ? ORDER BY doc_id",
            (character_name,),
        ).fetchall()
    return [Document.from_dict(dict(r)) for r in rows]


def delete_documents(character_name: str) -> int:
    """删除某人物的所有文档。返回删除条数。"""
    with get_conn() as conn:
        cur = conn.execute(
            "DELETE FROM documents WHERE character_name = ?", (character_name,)
        )
        return cur.rowcount


# ==================== 对话历史 ====================

def save_message(
    character_name: str, role: str, content: str, mode: str = "normal"
) -> None:
    """保存一条对话消息。"""
    with get_conn() as conn:
        conn.execute(
            """
            INSERT INTO conversations (character_name, role, content, mode)
            VALUES (?, ?, ?, ?)
            """,
            (character_name, role, content, mode),
        )


def get_history(
    character_name: str, limit: int = 20
) -> list[dict[str, str]]:
    """
    读取某人物的最近对话历史，返回 OpenAI messages 格式：
      [{"role": "user"/"assistant", "content": "..."}]
    """
    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT role, content FROM conversations
            WHERE character_name = ?
            ORDER BY id DESC LIMIT ?
            """,
            (character_name, limit),
        ).fetchall()
    # 反转为时间正序
    history = [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]
    return history


def clear_history(character_name: str) -> int:
    """清空某人物的对话历史。"""
    with get_conn() as conn:
        cur = conn.execute(
            "DELETE FROM conversations WHERE character_name = ?", (character_name,)
        )
        return cur.rowcount
