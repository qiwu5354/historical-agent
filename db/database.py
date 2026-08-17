"""
SQLite 数据库初始化与连接管理。

表结构：
  characters  - 人物档案（JSON 序列化存储）
  documents   - 知识库文档段落
  conversations - 对话历史
"""
from __future__ import annotations

import logging
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from config import DB_PATH, ensure_dirs

logger = logging.getLogger(__name__)


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS characters (
    name           TEXT PRIMARY KEY,
    profile_json   TEXT NOT NULL,
    created_at     TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS documents (
    doc_id          TEXT PRIMARY KEY,
    character_name  TEXT NOT NULL,
    source_type     TEXT NOT NULL,
    source_detail   TEXT,
    title           TEXT,
    era             TEXT,
    language        TEXT,
    content         TEXT NOT NULL,
    created_at      TEXT DEFAULT (datetime('now', 'localtime')),
    FOREIGN KEY (character_name) REFERENCES characters(name)
);
CREATE INDEX IF NOT EXISTS idx_documents_character ON documents(character_name);
CREATE INDEX IF NOT EXISTS idx_documents_source ON documents(source_type);

CREATE TABLE IF NOT EXISTS conversations (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    character_name  TEXT NOT NULL,
    role            TEXT NOT NULL,        -- 'user' | 'assistant'
    content         TEXT NOT NULL,
    mode            TEXT,                 -- 'normal' | 'socratic'
    created_at      TEXT DEFAULT (datetime('now', 'localtime')),
    FOREIGN KEY (character_name) REFERENCES characters(name)
);
CREATE INDEX IF NOT EXISTS idx_conversations_character ON conversations(character_name);
"""


def init_db(db_path: Path | None = None) -> None:
    """初始化数据库（创建表）。幂等，可重复调用。"""
    ensure_dirs()
    path = db_path or DB_PATH
    with sqlite3.connect(path) as conn:
        conn.execute("PRAGMA foreign_keys = ON")
        conn.executescript(SCHEMA_SQL)
        conn.commit()
    logger.info("数据库初始化完成: %s", path)


@contextmanager
def get_conn(db_path: Path | None = None) -> Iterator[sqlite3.Connection]:
    """获取数据库连接的上下文管理器。"""
    path = db_path or DB_PATH
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.row_factory = sqlite3.Row  # 支持按列名访问
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# 模块加载时自动初始化（延迟到首次使用更稳妥，这里做兜底）
def ensure_db() -> None:
    """确保数据库已初始化。在应用启动时调用。"""
    try:
        init_db()
    except Exception as e:
        logger.warning("数据库初始化失败（将在首次写入时重试）: %s", e)
