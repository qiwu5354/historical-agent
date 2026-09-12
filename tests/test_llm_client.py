"""
LLM 客户端回归测试。

覆盖曾经导致「HTTP 200 但 content 为空 -> JSON 解析失败」的那条链路：
  1. 思考开关按端点方言生成（DashScope / DeepSeek / 未知端点）；
  2. 「只输出 JSON」指令并入首条 system 消息，且不修改原 messages；
  3. 空正文 + 截断（finish_reason=length）时自动加大 max_tokens 并追加 /no_think 重试；
  4. 端点拒绝思考开关参数时自动去掉参数重试；
  5. 容错解析：代码块包裹、前后夹带解释、被截断的 JSON、空内容明确报错。

不依赖网络，全部用假响应对象驱动。
运行：python -m pytest tests -q
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import settings  # noqa: E402
from core.llm_client import (  # noqa: E402
    LLMClient,
    LLMError,
    _extract_content,
    _parse_json_lenient,
    _with_json_instruction,
)


# ==================== 假响应对象 ====================

def make_response(content: str, *, finish_reason="stop", reasoning: str = "",
                  completion_tokens: int = 10, reasoning_tokens=None):
    """构造一个形状与 openai SDK 响应一致的假对象。"""
    message = SimpleNamespace(content=content, reasoning_content=reasoning or None)
    usage = SimpleNamespace(
        completion_tokens=completion_tokens,
        completion_tokens_details=SimpleNamespace(reasoning_tokens=reasoning_tokens),
    )
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message, finish_reason=finish_reason)],
        usage=usage,
    )


class FakeCompletions:
    """记录每次调用参数，并按脚本依次返回响应（或抛异常）。"""

    def __init__(self, script):
        self.script = list(script)
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        item = self.script.pop(0) if self.script else make_response('{"ok": true}')
        if isinstance(item, Exception):
            raise item
        return item


def make_client(monkeypatch, script, *, base_url=None, model=None, disable_thinking=True):
    """构造一个打了桩的 LLMClient（不发真实请求）。"""
    monkeypatch.setattr(settings.llm, "api_key", "test-key", raising=False)
    monkeypatch.setattr(
        settings.llm, "base_url",
        base_url or settings.llm.base_url, raising=False,
    )
    monkeypatch.setattr(settings.llm, "model", model or settings.llm.model, raising=False)
    monkeypatch.setattr(settings.llm, "disable_thinking", disable_thinking, raising=False)
    client = LLMClient()
    fake = FakeCompletions(script)
    client._client = SimpleNamespace(chat=SimpleNamespace(completions=fake))
    return client, fake


DASHSCOPE = "https://dashscope.aliyuncs.com/compatible-mode/v1"
MESSAGES = [
    {"role": "system", "content": "你是研究助手。"},
    {"role": "user", "content": "提取档案。"},
]


# ==================== 1. 思考开关方言 ====================

def test_dashscope_uses_enable_thinking_false(monkeypatch):
    client, _ = make_client(monkeypatch, [make_response("{}")], base_url=DASHSCOPE)
    assert client._no_thinking_body == {"enable_thinking": False}


def test_deepseek_official_uses_thinking_disabled(monkeypatch):
    client, _ = make_client(
        monkeypatch, [make_response("{}")], base_url="https://api.deepseek.com"
    )
    assert client._no_thinking_body == {"thinking": {"type": "disabled"}}


def test_unknown_endpoint_sends_nothing(monkeypatch):
    client, _ = make_client(
        monkeypatch, [make_response("{}")], base_url="https://my-gateway.internal/v1"
    )
    assert client._no_thinking_body is None


def test_switch_can_be_turned_off(monkeypatch):
    client, _ = make_client(
        monkeypatch, [make_response("{}")], base_url=DASHSCOPE, disable_thinking=False
    )
    assert client._no_thinking_body is None


def test_no_think_suffix_only_for_self_hosted_qwen3(monkeypatch):
    client, _ = make_client(
        monkeypatch, [make_response("{}")],
        base_url="http://localhost:8000/v1", model="Qwen3-32B",
    )
    assert client._needs_no_think_suffix() is True

    client2, _ = make_client(
        monkeypatch, [make_response("{}")], base_url=DASHSCOPE, model="qwen3.7-flash"
    )
    assert client2._needs_no_think_suffix() is False


# ==================== 2. JSON 指令位置 ====================

def test_json_instruction_merged_into_first_system_message():
    out = _with_json_instruction(list(MESSAGES))
    assert len(out) == 2
    assert out[0]["role"] == "system"
    assert "只输出一个合法的 JSON" in out[0]["content"]
    assert "你是研究助手。" in out[0]["content"]
    assert out[1] == MESSAGES[1]


def test_json_instruction_does_not_mutate_input():
    original = [dict(m) for m in MESSAGES]
    _with_json_instruction(MESSAGES)
    assert MESSAGES == original


def test_json_instruction_inserted_when_no_system():
    out = _with_json_instruction([{"role": "user", "content": "hi"}])
    assert out[0]["role"] == "system"
    assert out[1]["role"] == "user"


# ==================== 3. 空正文自动恢复（核心回归） ====================

def test_empty_content_triggers_bigger_budget_retry(monkeypatch):
    """复现线上故障：第一次思考吃光预算 -> content 为空；重试应成功。"""
    truncated = make_response("", finish_reason="length", reasoning="想" * 2841,
                              completion_tokens=2048, reasoning_tokens=1990)
    ok = make_response(json.dumps({"name": "马克思"}, ensure_ascii=False))
    client, fake = make_client(monkeypatch, [truncated, ok], base_url=DASHSCOPE)
    monkeypatch.setattr(settings.llm, "max_tokens", 2048, raising=False)

    data = client.chat_json(MESSAGES)

    assert data == {"name": "马克思"}
    assert len(fake.calls) == 2
    assert fake.calls[0]["max_tokens"] == 2048
    assert fake.calls[1]["max_tokens"] == 4096  # 预算翻倍
    # 两次都带上了关闭思考的参数
    assert fake.calls[1]["extra_body"] == {"enable_thinking": False}
    # 第二次重试追加了 /no_think 提示词开关
    assert "/no_think" in fake.calls[1]["messages"][-1]["content"]


def test_retry_budget_is_capped(monkeypatch):
    truncated = make_response("", finish_reason="length", reasoning="想" * 100)
    client, fake = make_client(monkeypatch, [truncated, truncated], base_url=DASHSCOPE)
    monkeypatch.setattr(settings.llm, "max_tokens", 12288, raising=False)

    with pytest.raises(LLMError):
        client.chat_json(MESSAGES)
    assert fake.calls[1]["max_tokens"] == 16384  # 上限


def test_double_empty_raises_actionable_error(monkeypatch):
    truncated = make_response("", finish_reason="length", reasoning="想" * 2841)
    client, _ = make_client(monkeypatch, [truncated, truncated], base_url=DASHSCOPE)

    with pytest.raises(LLMError) as ei:
        client.chat_json(MESSAGES)

    msg = str(ei.value)
    assert "空正文" in msg
    assert "length" in msg
    assert "LLM_DISABLE_THINKING" in msg
    assert "LLM_MAX_TOKENS" in msg


def test_no_retry_when_thinking_already_enabled(monkeypatch):
    """用户主动开思考模式时不自动重试，直接把诊断信息抛出来。"""
    truncated = make_response("", finish_reason="length", reasoning="想" * 10)
    client, fake = make_client(
        monkeypatch, [truncated, truncated], base_url=DASHSCOPE, disable_thinking=False
    )
    with pytest.raises(LLMError):
        client.chat_json(MESSAGES)
    assert len(fake.calls) == 1


# ==================== 4. 参数不被支持时回退 ====================

class BadRequest(Exception):
    def __init__(self, msg):
        super().__init__(msg)
        self.status_code = 400


def test_falls_back_when_endpoint_rejects_thinking_param(monkeypatch):
    err = BadRequest("Error code: 400 - unknown parameter: enable_thinking")
    ok = make_response('{"name": "ok"}')
    client, fake = make_client(monkeypatch, [err, ok], base_url=DASHSCOPE)

    data = client.chat_json(MESSAGES)

    assert data == {"name": "ok"}
    assert "extra_body" in fake.calls[0]
    assert "extra_body" not in fake.calls[1]  # 已去掉该参数
    assert len(fake.calls) == 2


def test_unrelated_400_is_not_swallowed(monkeypatch):
    err = BadRequest("Error code: 400 - invalid api key")
    client, fake = make_client(monkeypatch, [err, make_response("{}")], base_url=DASHSCOPE)

    with pytest.raises(LLMError):
        client.chat_json(MESSAGES)
    assert len(fake.calls) == 1


# ==================== 5. 容错解析 ====================

@pytest.mark.parametrize(
    "raw",
    [
        '{"name": "马克思"}',
        '```json\n{"name": "马克思"}\n```',
        '```\n{"name": "马克思"}\n```',
        '好的，结果如下：\n{"name": "马克思"}\n希望有帮助。',
        '{"name": "马克思", "era": "1818-1883"}\n\n（以上为提取结果）',
    ],
)
def test_parse_tolerates_wrappers(raw):
    assert _parse_json_lenient(raw)["name"] == "马克思"


def test_parse_repairs_truncated_json():
    raw = '{"name": "马克思", "core_thought": "他的核心思想是唯'
    data = _parse_json_lenient(raw)
    assert data["name"] == "马克思"


def test_parse_repairs_truncated_nested_json():
    raw = '{"name": "马克思", "categories": ["哲学家", "经济学'
    data = _parse_json_lenient(raw)
    assert data["name"] == "马克思"


def test_parse_keeps_braces_inside_strings_intact():
    raw = '前言 {"name": "马克思", "note": "公式 {a} 与 [b]"} 后记'
    assert _parse_json_lenient(raw)["note"] == "公式 {a} 与 [b]"


def test_parse_unwraps_top_level_array():
    assert _parse_json_lenient('[{"name": "马克思"}]')["name"] == "马克思"


def test_parse_empty_content_message_is_explicit():
    with pytest.raises(LLMError) as ei:
        _parse_json_lenient("   ")
    assert "空内容" in str(ei.value)


def test_parse_garbage_still_fails_loudly():
    with pytest.raises(LLMError):
        _parse_json_lenient("模型今天不想干活")


# ==================== 6. 诊断信息提取 ====================

def test_extract_content_reports_reasoning_and_finish_reason():
    resp = make_response("", finish_reason="length", reasoning="想" * 50)
    content, info = _extract_content(resp, max_tokens=2048)
    assert content == ""
    assert info["finish_reason"] == "length"
    assert info["reasoning_chars"] == 50
    assert info["max_tokens"] == 2048


def test_extract_content_falls_back_to_reasoning_tokens():
    resp = make_response("正文", completion_tokens=99, reasoning_tokens=77)
    content, info = _extract_content(resp, max_tokens=4096)
    assert content == "正文"
    assert info["reasoning_chars"] == 77
