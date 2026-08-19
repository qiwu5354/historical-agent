"""
学者提示词增强器（ScholarEnricher）。

把以下要素融合为一段连贯的 System Prompt 增量：
  1. 主领域模板（术语/方法论/认识论/引用/语气）
  2. 副领域模板（压缩为参照片段，不削弱主领域风格）
  3. 职业阶段修饰词
  4. 方法论偏好选择（多选叠加，去重并保持一致）
  5. 沟通风格修饰词
  6. 研究焦点（自由文本注入，提醒聚焦特定问题）
  7. 引用密度控制
  8. 显式身份一致性约束
  9. 跨领域一致性原则（保证多领域学者语调不被削弱）

输出可作为 System Prompt 的「学者身份增强块」拼接到 character_prompt 中。
"""
from __future__ import annotations

from typing import Iterable

from prompts.career_stages import render_career_stage
from prompts.communication_styles import render_communication_style
from prompts.domain_templates import render_domains
from prompts.scholar_profile import (
    METHODOLOGICAL_APPROACHES,
    ScholarProfile,
    describe_profile,
)


# 引用密度对应的硬性指令
CITATION_DENSITY_INSTRUCTIONS: dict[str, str] = {
    "sparse": (
        "引用密度：稀疏。仅在关键论断处标注出处，避免每段都引。"
        "更重视人物本人的分析而非罗列文献。"
    ),
    "moderate": (
        "引用密度：适中。重要论断必须可追溯至材料或人物原作；"
        "一般性背景叙述无需逐句引。"
    ),
    "dense": (
        "引用密度：密集。每个非显然论断都应给出可追溯依据，"
        "优先引用【参考资料】原文段落，必要时引用人物原作章节。"
    ),
}


# 跨领域一致性原则（多领域学者必读）
_INTERDISCIPLINARY_CONSISTENCY = """### 跨领域一致性原则（多领域学者必读）
当人物横跨多个领域时，按以下原则保持身份与语调一致：
1. **主领域优先**：整体语调、术语体系、引用规范以主领域为准。
2. **副领域作为参照视角**：在主领域不擅长的问题上，可借助副领域的方法论与
   认识论作为补充视角，但不要让副领域术语喧宾夺主。
3. **避免风格漂移**：不要在一段回答中前后风格突变（如前半段哲学诠释、
   后半段工程务实），如确需切换，须明确过渡并说明理由。
4. **承认边界**：当问题超出该人物真实跨域范围时，明说「这超出我的直接经验」，
   而不是装作该领域的专家。
"""


def render_methodological_preferences(methods: Iterable[str]) -> str:
    """把方法论偏好渲染为提示词块。"""
    method_list = [m for m in methods if m in METHODOLOGICAL_APPROACHES]
    if not method_list:
        return ""
    lines = ["### 方法论偏好"]
    for m in method_list:
        lines.append(f"- {METHODOLOGICAL_APPROACHES[m]}")
    lines.append(
        "在人物本人方法论与之兼容时，按以上偏好组织论证；"
        "若偏好与人物史实立场冲突，以人物立场为准。"
    )
    return "\n".join(lines)


def render_research_focus(focus: str) -> str:
    """把研究焦点渲染为提示词块。"""
    focus = (focus or "").strip()
    if not focus:
        return ""
    return (
        "### 研究焦点\n"
        f"当问题与以下焦点相关时，优先展开：{focus}\n"
        "若问题与焦点无关，仍按人物本人完整思想回答，不必强行牵连。"
    )


def render_citation_density(density: str) -> str:
    return CITATION_DENSITY_INSTRUCTIONS.get(
        density, CITATION_DENSITY_INSTRUCTIONS["moderate"]
    )


def render_explicit_constraints(text: str) -> str:
    text = (text or "").strip()
    if not text:
        return ""
    return f"### 显式身份一致性约束\n{text}"


def enrich_scholar_prompt(profile: ScholarProfile) -> str:
    """
    把 ScholarProfile 渲染为一段连贯的 System Prompt 增量文本。

    各模块的拼接顺序设计为「从范式到行为」的递进，便于 LLM 在长上下文中
    保持注意力：领域范式 → 职业阶段 → 方法偏好 → 研究焦点 → 沟通风格 →
    引用密度 → 跨域一致性 → 显式约束。
    """
    blocks: list[str] = []

    # 0. 档案概要（让 LLM 一眼看到当前设定的关键参数）
    blocks.append(f"### 学者档案设定\n{describe_profile(profile)}")

    # 1. 主领域 + 副领域模板
    blocks.append(render_domains(profile.primary_domain, profile.secondary_domains))

    # 2. 跨领域一致性原则（仅在多领域时插入）
    if profile.secondary_domains:
        blocks.append(_INTERDISCIPLINARY_CONSISTENCY)

    # 3. 职业阶段
    blocks.append(render_career_stage(profile.career_stage))

    # 4. 方法论偏好
    method_block = render_methodological_preferences(profile.methodological_approach)
    if method_block:
        blocks.append(method_block)

    # 5. 研究焦点
    focus_block = render_research_focus(profile.research_focus)
    if focus_block:
        blocks.append(focus_block)

    # 6. 沟通风格
    blocks.append(render_communication_style(profile.communication_style))

    # 7. 引用密度
    blocks.append(render_citation_density(profile.citation_density))

    # 8. 显式身份一致性约束
    constraint_block = render_explicit_constraints(
        profile.explicit_consistency_constraints
    )
    if constraint_block:
        blocks.append(constraint_block)

    return "\n\n".join(blocks)


def enrich_with_character(
    profile: ScholarProfile | None,
    *,
    fallback_categories: Iterable[str] | None = None,
) -> str:
    """
    便捷入口：
      - 有 profile 时：渲染完整增强块。
      - 无 profile 但有 fallback_categories：从类别自动推导一个默认 profile。
      - 全部为空：返回空串（让调用方使用旧逻辑）。
    """
    if profile is not None:
        return enrich_scholar_prompt(profile)

    if fallback_categories:
        # 从旧类别推导默认 profile（mid + technical + moderate）
        derived = ScholarProfile.from_categories(fallback_categories)
        if derived.primary_domain:
            return enrich_scholar_prompt(derived)

    return ""
