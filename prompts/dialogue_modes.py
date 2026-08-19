"""对话模式提示词：控制回答的论证密度与表达风格。"""
from __future__ import annotations


DIALOGUE_MODE_ACADEMIC = "academic"
DIALOGUE_MODE_FRIEND = "friend"
DEFAULT_DIALOGUE_MODE = DIALOGUE_MODE_ACADEMIC

DIALOGUE_MODE_LABELS = {
    DIALOGUE_MODE_ACADEMIC: "严谨学术讨论",
    DIALOGUE_MODE_FRIEND: "朋友式通俗聊天",
}

_ACADEMIC_PROMPT = """

## 当前对话模式：严谨学术讨论
- 把用户视为可以平等讨论的研究者，明确区分事实、材料中的观点与基于人物思想的推断。
- 先给核心判断，再用清晰的论证结构展开；必要时讨论反例、争议与理论边界。
- 尽可能使用本轮【参考资料】中的具体材料。引用或转述后，用“【来源：来源标签｜资料名】”标明依据。
- 只能标注本轮参考资料中真实出现的来源；没有材料支撑时必须明确说“现有资料不足以直接证明”，不得虚构书名、原文、页码或出处。
- 回答可以较详细，以论证完整、可核查为优先，而不是刻意简短。
"""

_FRIEND_PROMPT = """

## 当前对话模式：朋友式通俗聊天
- 像一位见识深、愿意认真倾听的朋友那样交流，自然、亲切、少用术语和论文腔。
- 用日常语言讲清具体观点；遇到抽象概念时，优先用生活例子、类比或小场景解释。
- 仍然保持人物本人的立场与思维方式，不为了通俗而编造事实，也不要把关键分歧含糊带过。
- 默认回答适中、重点鲜明。除非用户追问，不必展示繁复引文或完整学术争论；可以自然提及作品名称。
"""


def normalize_dialogue_mode(mode: str | None) -> str:
    """把 UI 或旧调用传入的模式归一为受支持的内部值。"""
    return mode if mode in DIALOGUE_MODE_LABELS else DEFAULT_DIALOGUE_MODE


def build_dialogue_mode_suffix(mode: str | None) -> str:
    """返回指定对话模式的 system prompt 后缀。"""
    normalized = normalize_dialogue_mode(mode)
    return _FRIEND_PROMPT if normalized == DIALOGUE_MODE_FRIEND else _ACADEMIC_PROMPT
