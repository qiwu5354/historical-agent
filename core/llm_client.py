"""
LLM 客户端封装（OpenAI 兼容接口）。

通过 openai SDK 接入通义千问 / DeepSeek / 智谱 GLM 等国产模型，提供三种调用方式：
  - chat()           普通调用，返回完整字符串
  - stream_chat()    流式调用，逐 chunk 产出（用于 UI 实时输出）
  - chat_json()      结构化输出，返回解析后的 dict（用于档案提取）

关于「思考模式」（重要）：
  qwen3.5+ / qwen3.7-flash / qwen3.8 系列、GLM-5 系列等混合思考模型的**思考默认开启**，
  且思考内容（reasoning_content）与正文（content）**共享同一个 max_tokens 预算**。
  当 max_tokens 偏小（如 2048）而任务要求长输出（如 500 字 core_thought）时，思考会把
  预算吃光，结果 HTTP 200 但 message.content 为空字符串——表现为
  `json.loads("") -> Expecting value: line 1 column 1 (char 0)`。

  因此本模块默认关闭思考模式（见 config.LLMConfig.disable_thinking / LLM_DISABLE_THINKING）：
    - DashScope（通义千问）: extra_body={"enable_thinking": False}
    - DeepSeek 官方端点:      extra_body={"thinking": {"type": "disabled"}}
    - 其它第三方端点:         不发送该参数（避免 400），仅靠 max_tokens 余量兜底
  若端点不支持该参数，会自动去掉参数重试；若仍返回空正文，会自动加大 max_tokens 重试一次。
"""
from __future__ import annotations

import json
import logging
import re
import threading
from typing import Any, Iterator

from openai import OpenAI

from config import settings

logger = logging.getLogger(__name__)

# 出现这些字样，说明端点不认识我们发的扩展参数（思考开关），去掉后重试
_UNSUPPORTED_PARAM_HINTS = (
    "enable_thinking",
    "thinking",
    "unknown parameter",
    "unsupported parameter",
    "unrecognized",
    "extra fields not permitted",
    "invalid parameter",
)

# 第二次重试时 max_tokens 的放大上限，避免直接把上下文撑爆
_MAX_TOKENS_RETRY_CEILING = 16384

# 提取「首个完整 JSON 对象」用（应对模型在 JSON 前后夹带解释文字）
_JSON_START = re.compile(r"[{\[]")


class LLMError(RuntimeError):
    """LLM 调用相关错误"""


def _is_unsupported_param_error(exc: Exception) -> bool:
    """判断异常是否为「端点不支持该扩展参数」。"""
    msg = str(exc).lower()
    if not any(hint in msg for hint in _UNSUPPORTED_PARAM_HINTS):
        return False
    # 只在客户端错误（400/422）上触发回退，避免把限流/超时误判成参数问题
    status = getattr(exc, "status_code", None)
    return status is None or status in (400, 422)


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
        self._base_url = cfg.base_url or ""
        # 关闭思考模式的参数（None 表示该端点不支持 / 已关闭该特性）
        self._no_thinking_body: dict[str, Any] | None = (
            self._thinking_disabled_body() if cfg.disable_thinking else None
        )

    # ---------- 思考模式适配 ----------
    def _thinking_disabled_body(self) -> dict[str, Any] | None:
        """
        按端点方言生成「关闭思考」的 extra_body。
        只对已知支持该参数的厂商发送，未知端点返回 None（不冒险发参数）。
        """
        url = self._base_url.lower()
        if "dashscope" in url or "aliyuncs" in url:
            # 通义千问 / Model Studio：混合思考模式用 enable_thinking 逐请求开关
            return {"enable_thinking": False}
        if "deepseek" in url:
            # DeepSeek 官方端点
            return {"thinking": {"type": "disabled"}}
        logger.debug("未知端点 %s，不发送思考模式开关", self._base_url)
        return None

    def _needs_no_think_suffix(self) -> bool:
        """
        是否需要在提示词末尾追加 `/no_think`。
        用于开源的 Qwen3 混合思考模型（vLLM/SGLang 等自建端点不认 enable_thinking，
        但认提示词后缀）。
        """
        if not settings.llm.disable_thinking:
            return False
        if "dashscope" in self._base_url.lower():
            return False
        return "qwen3" in self._model.lower()

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
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "temperature": cfg.temperature if temperature is None else temperature,
            "max_tokens": cfg.max_tokens if max_tokens is None else max_tokens,
            "timeout": timeout if timeout is not None else cfg.stream_timeout,
        }
        if self._no_thinking_body:
            kwargs["extra_body"] = self._no_thinking_body
        try:
            resp = self._create(kwargs)
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
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "temperature": cfg.temperature if temperature is None else temperature,
            "max_tokens": cfg.max_tokens if max_tokens is None else max_tokens,
            "stream": True,
            "timeout": cfg.stream_timeout,
        }
        if self._no_thinking_body:
            kwargs["extra_body"] = self._no_thinking_body
        try:
            stream = self._create(kwargs)
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
        要求模型输出 JSON 并解析为 dict。用于人物识别、档案提取等结构化任务。

        容错链：
          1. 把「只输出 JSON」指令并入**第一条 system 消息**头部
             （不再在 user 之后追加 system——部分模型会忽略位置异常的第二条 system）；
          2. 关闭思考模式，避免思考 token 挤占正文预算；
          3. 正文为空且是被 max_tokens 截断时，自动加大 max_tokens 并追加 /no_think 重试一次；
          4. 端点不认思考开关时，自动去掉该参数重试；
          5. 解析做容错：剥代码块、截取首个 JSON 对象、必要时补全被截断的括号。
        """
        cfg = settings.llm
        temp = cfg.serious_temperature if temperature is None else temperature
        instructed = _with_json_instruction(messages)

        budget = cfg.max_tokens
        call_messages = instructed
        raw, info = self._json_call(call_messages, temp, budget, timeout)

        # 空正文兜底：多半是思考把 max_tokens 吃光了（finish_reason=length）
        if not raw.strip() and cfg.disable_thinking:
            bigger = min(max(budget * 2, 4096), _MAX_TOKENS_RETRY_CEILING)
            logger.warning(
                "LLM 返回空正文（finish_reason=%s，思考约 %d 字符），"
                "将 max_tokens 提升到 %d 并追加 /no_think 重试一次",
                info.get("finish_reason"),
                info.get("reasoning_chars") or 0,
                bigger,
            )
            call_messages = _with_no_think_suffix(instructed)
            raw, info = self._json_call(call_messages, temp, bigger, timeout)

        if not raw.strip():
            raise LLMError(
                "模型返回了空正文（HTTP 200 但 content 为空），无法解析为 JSON。"
                f"finish_reason={info.get('finish_reason')!r}，"
                f"思考内容 {info.get('reasoning_chars') or 0} 字符，"
                f"计费输出 {info.get('completion_tokens')} token，"
                f"max_tokens={info.get('max_tokens')}，模型={self._model}。\n"
                "常见原因：该模型默认开启「思考模式」，思考 token 与正文共享 max_tokens 预算。\n"
                "建议：① 保持 LLM_DISABLE_THINKING=1；② 把 LLM_MAX_TOKENS 调到 4096 以上；"
                "③ 若用的是「纯思考模型」（如 qwen3-*-thinking / deepseek-r1），请换成混合思考模型。"
            )

        if info.get("finish_reason") == "length":
            logger.warning(
                "模型输出被 max_tokens 截断（max_tokens=%s，正文 %d 字符），"
                "JSON 可能不完整，已尝试补全。建议调大 LLM_MAX_TOKENS。",
                info.get("max_tokens"),
                len(raw),
            )

        return _parse_json_lenient(raw)

    # ---------- 内部：发起一次请求 ----------
    def _create(self, kwargs: dict[str, Any]):
        """
        调用 chat.completions.create，并在端点拒绝思考开关参数时自动去掉参数重试。
        """
        try:
            return self._client.chat.completions.create(**kwargs)
        except Exception as e:
            if "extra_body" in kwargs and _is_unsupported_param_error(e):
                logger.warning(
                    "端点不接受思考模式开关参数（%s），已去掉该参数重试", self._base_url
                )
                retry_kwargs = dict(kwargs)
                retry_kwargs.pop("extra_body", None)
                return self._client.chat.completions.create(**retry_kwargs)
            raise

    def _json_call(
        self,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
        timeout: float | None,
    ) -> tuple[str, dict[str, Any]]:
        """发起一次 JSON 调用，返回 (正文, 诊断信息)。"""
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "timeout": timeout if timeout is not None else settings.llm.stream_timeout,
        }
        if self._no_thinking_body:
            kwargs["extra_body"] = self._no_thinking_body
        try:
            resp = self._create(kwargs)
        except Exception as e:
            logger.exception("LLM chat_json 调用失败")
            raise LLMError(f"LLM 调用失败: {e}") from e
        return _extract_content(resp, max_tokens)


# ==================== 内部工具 ====================

_JSON_INSTRUCTION = (
    "请严格只输出一个合法的 JSON 对象，不要任何解释性文字、"
    "不要 markdown 代码块标记。所有字符串值用中文。"
)


def _with_json_instruction(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    """
    把「只输出 JSON」的约束并入第一条 system 消息。
    若首条不是 system，则在最前面插入一条；不修改原 messages。
    """
    out = [dict(m) for m in messages]
    if out and out[0].get("role") == "system":
        out[0]["content"] = f"{_JSON_INSTRUCTION}\n\n{out[0].get('content', '')}"
        return out
    return [{"role": "system", "content": _JSON_INSTRUCTION}, *out]


def _with_no_think_suffix(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    """给最后一条 user 消息追加 `/no_think`（开源 Qwen3 混合思考模型的提示词开关）。"""
    out = [dict(m) for m in messages]
    for m in reversed(out):
        if m.get("role") == "user":
            m["content"] = f"{m.get('content', '')}\n\n/no_think"
            break
    return out


def _extract_content(resp: Any, max_tokens: int) -> tuple[str, dict[str, Any]]:
    """
    从响应中取出正文，并汇总诊断信息（含思考内容长度）。
    """
    content = ""
    reasoning_chars: int | None = None
    finish_reason: str | None = None
    completion_tokens: int | None = None
    if getattr(resp, "choices", None):
        choice = resp.choices[0]
        finish_reason = getattr(choice, "finish_reason", None)
        message = getattr(choice, "message", None)
        if message is not None:
            content = getattr(message, "content", None) or ""
            reasoning = getattr(message, "reasoning_content", None)
            if reasoning:
                reasoning_chars = len(reasoning)
            else:
                # OpenAI SDK 默认丢弃未知字段，兜底从 model_extra 里找
                extra = getattr(message, "model_extra", None) or {}
                reasoning = extra.get("reasoning_content")
                if reasoning:
                    reasoning_chars = len(reasoning)
    usage = getattr(resp, "usage", None)
    if usage is not None:
        completion_tokens = getattr(usage, "completion_tokens", None)
        details = getattr(usage, "completion_tokens_details", None)
        reasoning_tokens = getattr(details, "reasoning_tokens", None) if details else None
        if reasoning_chars is None and reasoning_tokens:
            reasoning_chars = reasoning_tokens  # 拿不到原文时用 token 数近似
    return content, {
        "finish_reason": finish_reason,
        "reasoning_chars": reasoning_chars,
        "completion_tokens": completion_tokens,
        "max_tokens": max_tokens,
    }


def _strip_code_fence(text: str) -> str:
    """去掉 ```json ... ``` / ``` ... ``` 包裹。"""
    if not text.startswith("```"):
        return text
    body = text.split("\n", 1)[1] if "\n" in text else ""
    if body.rstrip().endswith("```"):
        body = body.rstrip()[:-3]
    return body.strip()


def _first_json_slice(text: str) -> str | None:
    """
    截取从第一个 `{` / `[` 开始、括号配平的片段。
    应对模型在 JSON 前后夹杂说明文字的情况；同时跳过字符串内部的括号。
    """
    start = _JSON_START.search(text)
    if not start:
        return None
    opening = text[start.start()]
    closing = "}" if opening == "{" else "]"
    depth = 0
    in_string = False
    escaped = False
    for i in range(start.start(), len(text)):
        ch = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == opening:
            depth += 1
        elif ch == closing:
            depth -= 1
            if depth == 0:
                return text[start.start(): i + 1]
    return None


def _repair_truncated_json(text: str) -> str | None:
    """尽力补全被 max_tokens 截断的 JSON：闭合未结束的字符串与括号。"""
    start = _JSON_START.search(text)
    if not start:
        return None
    body = text[start.start():]
    stack: list[str] = []
    in_string = False
    escaped = False
    for ch in body:
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch in "{[":
            stack.append("}" if ch == "{" else "]")
        elif ch in "}]" and stack:
            stack.pop()
    if not stack and not in_string:
        return None  # 不是截断，交给正常解析报错
    repaired = body.rstrip()
    if in_string:
        repaired += '"'
    # 去掉可能悬空的尾随逗号/冒号
    repaired = repaired.rstrip().rstrip(",").rstrip()
    if repaired.endswith(":"):
        repaired += ' ""'
    repaired += "".join(reversed(stack))
    return repaired


def _parse_json_lenient(raw: str) -> dict:
    """
    容错解析 JSON：
      1. 空内容早退，给出明确原因（而不是含糊的 JSON 报错）；
      2. 去掉 markdown 代码块包裹；
      3. 直接解析 -> 截取首个 JSON 对象 -> 补全被截断的括号，逐级降级；
      4. 上述都失败才抛错，并在日志里打印原始内容便于排查。
    """
    text = (raw or "").strip()
    if not text:
        raise LLMError(
            "模型返回空内容，无法解析为 JSON（HTTP 200 但 content 为空）。"
            "通常是模型「思考模式」把 max_tokens 预算耗尽，见 core/llm_client.py 顶部说明。"
        )

    text = _strip_code_fence(text)

    candidates: list[str] = [text]
    sliced = _first_json_slice(text)
    if sliced and sliced != text:
        candidates.append(sliced)
    repaired = _repair_truncated_json(text)
    if repaired:
        candidates.append(repaired)

    last_error: Exception | None = None
    for cand in candidates:
        try:
            data = json.loads(cand)
        except json.JSONDecodeError as e:
            last_error = e
            continue
        if isinstance(data, dict):
            if cand is not text:
                logger.warning("JSON 直接解析失败，已通过容错截取/补全成功解析")
            return data
        # 模型偶尔用数组包一层
        if isinstance(data, list) and data and isinstance(data[0], dict):
            return data[0]
        last_error = LLMError("模型输出的 JSON 顶层不是对象")
        break

    logger.error(
        "JSON 解析失败: %s\n原始内容(前500字): %r", last_error, text[:500]
    )
    raise LLMError(f"模型输出无法解析为 JSON: {last_error}") from last_error


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
