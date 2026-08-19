"""
LLM 客户端封装（OpenAI 兼容接口）。

通过 openai SDK 接入智谱 GLM / 通义千问 / DeepSeek 等国产模型，
提供三种调用方式：
  - chat()           普通调用，返回完整字符串
  - stream_chat()    流式调用，逐 chunk 产出（用于 UI 实时输出）
  - chat_json()      结构化输出，返回解析后的 dict（用于档案提取）
"""
from __future__ import annotations

import json
import logging
import threading
from typing import Iterator

from openai import OpenAI

from config import settings

logger = logging.getLogger(__name__)


class LLMError(RuntimeError):
    """LLM 调用相关错误"""


class LLMClient:
    """OpenAI 兼容接口的轻量封装"""

    def __init__(self) -> None:
        cfg = settings.llm
        if not cfg.api_key:
            raise LLMError(
                "未配置 LLM_API_KEY。请在 .env 中设置，参考 .env.example。"
            )
        self._client = OpenAI(api_key=cfg.api_key, base_url=cfg.base_url)
        self._model = cfg.model

    # ---------- 普通调用 ----------
    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        timeout: float | None = None,
    ) -> str:
        """普通调用，返回完整回复文本。"""
        cfg = settings.llm
        try:
            resp = self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                temperature=cfg.temperature if temperature is None else temperature,
                max_tokens=cfg.max_tokens if max_tokens is None else max_tokens,
                timeout=timeout if timeout is not None else cfg.stream_timeout,
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception as e:
            logger.exception("LLM chat 调用失败")
            raise LLMError(f"LLM 调用失败: {e}") from e

    # ---------- 流式调用 ----------
    def stream_chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> Iterator[str]:
        """流式调用，逐 token 产出文本片段。"""
        cfg = settings.llm
        try:
            stream = self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                temperature=cfg.temperature if temperature is None else temperature,
                max_tokens=cfg.max_tokens if max_tokens is None else max_tokens,
                stream=True,
                timeout=cfg.stream_timeout,
            )
            for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta
        except Exception as e:
            logger.exception("LLM stream_chat 调用失败")
            raise LLMError(f"LLM 流式调用失败: {e}") from e

    # ---------- 结构化 JSON 输出 ----------
    def chat_json(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float | None = None,
        timeout: float | None = None,
    ) -> dict:
        """
        要求模型输出 JSON 并解析为 dict。
        用于人物档案提取等结构化任务。

        实现：在 messages 中追加 JSON 指令，并做容错解析。
        """
        cfg = settings.llm
        temp = cfg.serious_temperature if temperature is None else temperature
        # 追加 JSON 输出要求（不修改原 messages）
        instructed = list(messages) + [
            {
                "role": "system",
                "content": (
                    "请严格只输出一个合法的 JSON 对象，不要任何解释性文字、"
                    "不要 markdown 代码块标记。所有字符串值用中文。"
                ),
            }
        ]
        try:
            resp = self._client.chat.completions.create(
                model=self._model,
                messages=instructed,
                temperature=temp,
                max_tokens=cfg.max_tokens,
                timeout=timeout if timeout is not None else cfg.stream_timeout,
            )
            raw = (resp.choices[0].message.content or "").strip()
        except Exception as e:
            logger.exception("LLM chat_json 调用失败")
            raise LLMError(f"LLM 调用失败: {e}") from e

        return _parse_json_lenient(raw)


def _parse_json_lenient(raw: str) -> dict:
    """容错解析 JSON：去除 markdown 代码块包裹后解析。"""
    text = raw.strip()
    # 去除可能的 ```json ... ``` 包裹
    if text.startswith("```"):
        # 去掉首行 ```xxx
        text = text.split("\n", 1)[-1] if "\n" in text else text
        # 去掉结尾 ```
        if text.endswith("```"):
            text = text.rsplit("```", 1)[0]
        text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        logger.error("JSON 解析失败: %s\n原始内容(前500字): %s", e, text[:500])
        raise LLMError(f"模型输出无法解析为 JSON: {e}") from e


# 全局单例（懒加载 + 双重检查锁，Gradio 多线程下避免产生多个实例）
_client_singleton: LLMClient | None = None
_client_lock = threading.Lock()


def get_llm() -> LLMClient:
    """获取全局 LLM 客户端单例（线程安全）。"""
    global _client_singleton
    if _client_singleton is None:
        with _client_lock:
            if _client_singleton is None:
                _client_singleton = LLMClient()
    return _client_singleton
