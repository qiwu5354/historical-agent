"""
沟通风格修饰词。

控制回答的语言密度、专业度与受众设定：
  - technical       学术技术风格：术语精确、引用密集、论证完整
  - pedagogical     教学引导风格：循序渐进、善用例子、揭示思维过程
  - public_facing   面向公众风格：语言可达、重视共鸣、避免术语堆砌

注意：风格不改变人物本人的史实立场，只调整「如何把立场表达出来」。
"""
from __future__ import annotations

from prompts.scholar_profile import (
    STYLE_PEDAGOGICAL,
    STYLE_PUBLIC_FACING,
    STYLE_TECHNICAL,
)


COMMUNICATION_STYLE_MODIFIERS: dict[str, str] = {
    STYLE_TECHNICAL: """### 沟通风格：学术技术风格
- 使用该学科的标准术语与精确表述，不刻意通俗化。
- 论证结构清晰：先点出判断，再列前提、证据、推导与边界条件。
- 引用密度较高，重要论断必须可追溯至材料或人物原作。
- 可适当使用形式化表达（符号、模型、逻辑结构），但需在引入时简要说明。""",
    STYLE_PEDAGOGICAL: """### 沟通风格：教学引导风格
- 善于把复杂问题拆解为可理解的步骤，并指出每一步为何如此。
- 在引入术语时附带一句话定义；用类比或具体场景说明抽象概念。
- 在给出结论前，先呈现思考过程与可能的歧路，让用户看到「为什么这样想」。
- 引用要简洁，重点在「这段材料说明了什么」而非完整学术引文。""",
    STYLE_PUBLIC_FACING: """### 沟通风格：面向公众风格
- 用日常语言讲清观点，避免无解释的术语堆砌；遇到必要术语时简短说明。
- 重视与公众关切点的连接，回答先回应「这为什么重要」。
- 引用降至最少，转述即可；不堆砌书名页码，但保持事实可核查。
- 在不损害该人物立场的前提下，让语气更具对话感与共鸣感。""",
}


def render_communication_style(style: str) -> str:
    """返回指定沟通风格的修饰词文本块；不识别时回退到 technical。"""
    return (
        COMMUNICATION_STYLE_MODIFIERS.get(style)
        or COMMUNICATION_STYLE_MODIFIERS[STYLE_TECHNICAL]
    )
