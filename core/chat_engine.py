"""
对话引擎：把人物档案、RAG 检索、历史记录、LLM 串联成完整对话流程。

对外提供：
  - chat()         普通对话（一次性返回完整回复）
  - stream_chat()  流式对话（逐 token 产出，用于 UI）
"""
from __future__ import annotations

import logging
from typing import Iterator

from config import settings
from core.llm_client import get_llm, LLMError
from core.rag_engine import build_context
from db import repository as repo
from models.character import Character
from prompts import safe_format
from prompts.character_prompt import build_system_prompt
from prompts.dialogue_modes import (
    DEFAULT_DIALOGUE_MODE,
    build_dialogue_mode_suffix,
    normalize_dialogue_mode,
)
from prompts.socratic_prompt import build_socratic_suffix

logger = logging.getLogger(__name__)

# 注入 RAG 上下文给 LLM 的引导语前缀
_RAG_USER_PREFIX = "【参考资料】以下是与本问题相关的、来自我（{name}）著作和资料的片段，请结合它们作答：\n{context}\n\n【我的问题】"


def _build_messages(
    character: Character,
    user_input: str,
    history: list[dict[str, str]],
    *,
    use_rag: bool = True,
    socratic: bool = False,
    dialogue_mode: str = DEFAULT_DIALOGUE_MODE,
) -> list[dict[str, str]]:
    """组装完整 messages 列表。"""
    # 1. System Prompt
    system = build_system_prompt(character)
    system += build_dialogue_mode_suffix(dialogue_mode)
    if socratic:
        system += build_socratic_suffix(character.name)
    messages: list[dict[str, str]] = [{"role": "system", "content": system}]

    # 2. 历史对话（取最近若干轮，控制 token）
    # 从最近的 user 消息起截，避免从 assistant 中段截断后 messages[1] 是 assistant，
    # 部分国产 OpenAI 兼容模型对此敏感会报错或行为异常。
    recent = history[-10:]
    # 若截断点落在 assistant 上，向前回退到第一个 user
    while recent and recent[0].get("role") != "user":
        recent = recent[1:]
    messages.extend(recent)

    # 3. RAG 检索增强（注入到本轮 user 消息）
    user_content = user_input
    if use_rag:
        try:
            context = build_context(character.name, user_input)
            if context:
                user_content = safe_format(
                    _RAG_USER_PREFIX, name=character.name, context=context
                ) + "\n" + user_input
        except Exception as e:
            logger.warning("RAG 上下文构建失败，继续无增强对话: %s", e)

    messages.append({"role": "user", "content": user_content})
    return messages


def chat(
    character: Character,
    user_input: str,
    history: list[dict[str, str]] | None = None,
    *,
    use_rag: bool = True,
    socratic: bool = False,
    dialogue_mode: str = DEFAULT_DIALOGUE_MODE,
) -> str:
    """普通对话：返回完整回复字符串。"""
    history = history or []
    dialogue_mode = normalize_dialogue_mode(dialogue_mode)
    messages = _build_messages(
        character,
        user_input,
        history,
        use_rag=use_rag,
        socratic=socratic,
        dialogue_mode=dialogue_mode,
    )
    llm = get_llm()
    reply = llm.chat(messages)
    # 持久化
    mode = f"{dialogue_mode}+socratic" if socratic else dialogue_mode
    repo.save_message(character.name, "user", user_input, mode)
    repo.save_message(character.name, "assistant", reply, mode)
    return reply


def stream_chat(
    character: Character,
    user_input: str,
    history: list[dict[str, str]] | None = None,
    *,
    use_rag: bool = True,
    socratic: bool = False,
    dialogue_mode: str = DEFAULT_DIALOGUE_MODE,
) -> Iterator[str]:
    """流式对话：逐 token 产出回复片段。"""
    history = history or []
    dialogue_mode = normalize_dialogue_mode(dialogue_mode)
    messages = _build_messages(
        character,
        user_input,
        history,
        use_rag=use_rag,
        socratic=socratic,
        dialogue_mode=dialogue_mode,
    )
    llm = get_llm()
    mode = f"{dialogue_mode}+socratic" if socratic else dialogue_mode
    # 先存用户消息
    repo.save_message(character.name, "user", user_input, mode)

    # 流式收集完整回复，结束后再存库
    collected: list[str] = []
    errored = False
    try:
        for chunk in llm.stream_chat(messages):
            collected.append(chunk)
            yield chunk
    except LLMError as e:
        # 出错时只向前端透出错误信息，不持久化进对话历史，
        # 避免下次重新加载历史时把错误消息当 assistant 回复带回对话上下文。
        errored = True
        yield f"\n\n[对话出错：{e}]"

    if not errored:
        repo.save_message(character.name, "assistant", "".join(collected), mode)
