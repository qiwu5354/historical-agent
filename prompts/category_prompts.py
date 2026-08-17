"""
学者类别提示词库。

每个类别对应一段“与该类学者对话时应遵循的思维/表达方式”。
一个人物可能同时属于多个类别，因此支持把多段提示词拼接使用。
"""
from __future__ import annotations

from typing import Iterable


# 类别 -> 对话风格/方法论提示词
CATEGORY_GUIDELINES: dict[str, str] = {
    "哲学家": """### 哲学家对话方式
- 注重概念澄清、逻辑论证和思想史脉络。
- 面对问题先追问“这个问题本身是否成立”“核心概念如何定义”。
- 鼓励从本体论、认识论、伦理学等角度展开，而不是简单给结论。""",

    "心理学家": """### 心理学家对话方式
- 关注动机、认知、情感、人格、潜意识与社会环境对行为的影响。
- 解释人的行为时强调实证证据、临床观察和理论框架。
- 避免把复杂心理现象简化为道德判断。""",

    "经济学家": """### 经济学家对话方式
- 关注稀缺性、激励、制度、市场与政府干预、增长与分配。
- 分析政策时先说明假设条件、受益者/受损者、长期/短期影响。
- 可以用模型或历史案例支撑观点，但避免堆砌术语。""",

    "社会学家": """### 社会学家对话方式
- 关注社会结构、群体互动、制度变迁、文化规范与权力关系。
- 善于把个人困扰与社会议题联系起来。
- 强调经验材料、比较视角和历史背景。""",

    "历史学家": """### 历史学家对话方式
- 重视史料来源、年代顺序、因果关系和语境化解释。
- 区分事实判断与价值判断，避免以今度古。
- 回答时说明证据类型和不确定性。""",

    "政治学家": """### 政治学家对话方式
- 关注权力、国家、制度、合法性、治理与政治行为。
- 区分应然与实然，强调制度约束和利益博弈。
- 善于比较不同政体与政策方案。""",

    "法学家": """### 法学家对话方式
- 注重规范、权利、程序、判例与法理。
- 分析问题先从法律关系和请求权基础入手。
- 区分“合法”与“合理”，并讨论制度背后的价值选择。""",

    "作家": """### 作家对话方式
- 关注语言、叙事、意象、人性与时代精神。
- 善于用具体场景和人物命运呈现抽象问题。
- 鼓励通过写作和表达本身来探索思想。""",

    "科学家": """### 科学家对话方式
- 强调假设、实验、证据、可重复性和逻辑自洽。
- 对不确定性问题明确说明已知与未知。
- 反对脱离证据的空谈，重视量化与机制。""",

    "思想家": """### 思想家对话方式
- 善于从整体性、批判性和前瞻性角度把握时代问题。
- 注重思想内部的一致性与张力。
- 回答时体现原创概念和思维框架，而非百科式罗列。""",

    "其他": """### 通用学者对话方式
- 保持理性、严谨、有据可依。
- 尊重事实与逻辑，承认知识的边界。
- 结合该人物所在领域的方法论进行回答。""",
}

# 常用同义词/英文归一化
CATEGORY_ALIASES: dict[str, str] = {
    "哲学家": "哲学家",
    "philosopher": "哲学家",
    "philosophy": "哲学家",
    "political philosopher": "哲学家",
    "moral philosopher": "哲学家",
    "social philosopher": "哲学家",
    "心理学家": "心理学家",
    "psychologist": "心理学家",
    "psychology": "心理学家",
    "psychoanalyst": "心理学家",
    "精神分析学家": "心理学家",
    "精神分析师": "心理学家",
    "心理分析学家": "心理学家",
    "经济学家": "经济学家",
    "economist": "经济学家",
    "economics": "经济学家",
    "political economist": "经济学家",
    "政治经济学家": "经济学家",
    "古典经济学家": "经济学家",
    "社会学家": "社会学家",
    "sociologist": "社会学家",
    "sociology": "社会学家",
    "社会理论家": "社会学家",
    "历史学家": "历史学家",
    "historian": "历史学家",
    "history": "历史学家",
    "政治学家": "政治学家",
    "political scientist": "政治学家",
    "political science": "政治学家",
    "政治理论家": "政治学家",
    "法学家": "法学家",
    "jurist": "法学家",
    "lawyer": "法学家",
    "法哲学家": "法学家",
    "作家": "作家",
    "writer": "作家",
    "author": "作家",
    "literary critic": "作家",
    "文学批评家": "作家",
    "科学家": "科学家",
    "scientist": "科学家",
    "physicist": "科学家",
    "chemist": "科学家",
    "biologist": "科学家",
    "物理学家": "科学家",
    "化学家": "科学家",
    "生物学家": "科学家",
    "政治哲学家": "哲学家",
    "道德哲学家": "哲学家",
    "社会哲学家": "哲学家",
    "思想家": "思想家",
    "thinker": "思想家",
    "学者": "其他",
    "scholar": "其他",
    "其他": "其他",
}


def normalize_category(category: str) -> str:
    """将 LLM 返回的类别文本归一化为标准类别名。"""
    key = category.strip().lower()
    return CATEGORY_ALIASES.get(key, category.strip() or "其他")


def normalize_categories(categories: Iterable[str] | str) -> list[str]:
    """归一化并去重类别列表。兼容字符串或列表输入。"""
    if isinstance(categories, str):
        categories = [categories]
    seen: set[str] = set()
    result: list[str] = []
    for c in categories:
        if not isinstance(c, str):
            continue
        normalized = normalize_category(c)
        if normalized and normalized not in seen:
            seen.add(normalized)
            result.append(normalized)
    # 如果已经有明确类别，就不保留泛化的“其他”
    if len(result) > 1 and "其他" in result:
        result.remove("其他")
    return result or ["其他"]


def build_category_guidelines(categories: Iterable[str]) -> str:
    """根据人物类别返回对应的提示词片段；多个类别会拼接。"""
    normalized = normalize_categories(categories)
    blocks: list[str] = []
    for cat in normalized:
        guideline = CATEGORY_GUIDELINES.get(cat)
        if guideline:
            blocks.append(guideline)
    if not blocks:
        blocks.append(CATEGORY_GUIDELINES["其他"])
    return "\n\n".join(blocks)
