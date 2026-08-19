"""
学者档案（ScholarProfile）—— 在角色档案之上叠加可调维度的「学者身份参数」。

设计意图：让同一历史人物可被配置为不同职业阶段、研究焦点、方法偏好与沟通风格，
从而生成身份一致但侧重不同的对话。例如「年轻时期的弗洛伊德」vs「晚年的弗洛伊德」，
「严谨学术的爱因斯坦」vs「面向公众的爱因斯坦」。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

from prompts.domain_templates import (
    ALL_DOMAINS,
    DOMAIN_LABELS,
    resolve_subdomain,
)


# ==================== 可调维度常量 ====================
CAREER_EARLY = "early_career"
CAREER_MID = "mid_career"
CAREER_SENIOR = "senior"

CAREER_LABELS: dict[str, str] = {
    CAREER_EARLY: "早期学者",
    CAREER_MID: "中期学者",
    CAREER_SENIOR: "资深/晚年学者",
}

# 方法论偏好（多选）
METHODOLOGICAL_APPROACHES: dict[str, str] = {
    "empirical": "实证/经验取向（重数据、观测、案例）",
    "theoretical": "理论/形式取向（重模型、推导、形式化）",
    "comparative": "比较取向（跨文化、跨时代、跨案例对比）",
    "historical": "历史取向（重语境、思想史、制度演变）",
    "experimental": "实验取向（重受控实验、可重复性）",
    "interpretive": "诠释取向（重文本细读、意义重建）",
    "critical": "批判取向（重权力分析、前提反思）",
    "applied": "应用取向（重可行解、干预效果、伦理边界）",
}

# 沟通风格
STYLE_TECHNICAL = "technical"
STYLE_PEDAGOGICAL = "pedagogical"
STYLE_PUBLIC_FACING = "public_facing"

STYLE_LABELS: dict[str, str] = {
    STYLE_TECHNICAL: "学术技术风格",
    STYLE_PEDAGOGICAL: "教学引导风格",
    STYLE_PUBLIC_FACING: "面向公众风格",
}


@dataclass
class ScholarProfile:
    """学者身份参数集。所有字段均为可选，缺失时用合理默认值填充。

    字段：
      primary_domain: 主领域（必填）。取值见 ALL_DOMAINS。
      secondary_domains: 副领域列表。跨学科学者在此声明。
      subdomain: 在主领域内的细分（如 "physics"、"economics"、"philosophy"）。
      career_stage: 职业阶段。影响对话的成熟度与立场。
      research_focus: 研究焦点的自由文本描述（如「意识的结构与功能」）。
      methodological_approach: 方法论偏好，列表，可为多个。
      communication_style: 沟通风格。
      citation_density: 引用密度偏好。"sparse" | "moderate" | "dense"
      explicit_consistency_constraints: 额外的显式身份一致性约束，自由文本。
    """
    primary_domain: str
    secondary_domains: list[str] = field(default_factory=list)
    subdomain: str = ""
    career_stage: str = CAREER_MID
    research_focus: str = ""
    methodological_approach: list[str] = field(default_factory=list)
    communication_style: str = STYLE_TECHNICAL
    citation_density: str = "moderate"
    explicit_consistency_constraints: str = ""

    # -------- 校验 --------
    def validate(self) -> list[str]:
        """返回校验错误信息列表；空列表表示通过。"""
        errors: list[str] = []
        if self.primary_domain not in ALL_DOMAINS:
            errors.append(
                f"primary_domain 不合法：{self.primary_domain}；"
                f"可选：{list(ALL_DOMAINS)}"
            )
        for d in self.secondary_domains:
            if d not in ALL_DOMAINS:
                errors.append(f"secondary_domains 含不合法项：{d}")
            if d == self.primary_domain:
                errors.append(f"secondary_domains 不应与 primary_domain 重复：{d}")
        if self.career_stage not in CAREER_LABELS:
            errors.append(
                f"career_stage 不合法：{self.career_stage}；"
                f"可选：{list(CAREER_LABELS.keys())}"
            )
        for m in self.methodological_approach:
            if m not in METHODOLOGICAL_APPROACHES:
                errors.append(f"methodological_approach 含不合法项：{m}")
        if self.communication_style not in STYLE_LABELS:
            errors.append(
                f"communication_style 不合法：{self.communication_style}；"
                f"可选：{list(STYLE_LABELS.keys())}"
            )
        if self.citation_density not in ("sparse", "moderate", "dense"):
            errors.append(f"citation_density 不合法：{self.citation_density}")
        return errors

    def is_valid(self) -> bool:
        return not self.validate()

    # -------- 序列化 --------
    def to_dict(self) -> dict[str, Any]:
        return {
            "primary_domain": self.primary_domain,
            "secondary_domains": list(self.secondary_domains),
            "subdomain": self.subdomain,
            "career_stage": self.career_stage,
            "research_focus": self.research_focus,
            "methodological_approach": list(self.methodological_approach),
            "communication_style": self.communication_style,
            "citation_density": self.citation_density,
            "explicit_consistency_constraints": self.explicit_consistency_constraints,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ScholarProfile":
        return cls(
            primary_domain=d.get("primary_domain") or "humanities",
            secondary_domains=list(d.get("secondary_domains") or []),
            subdomain=str(d.get("subdomain") or ""),
            career_stage=d.get("career_stage") or CAREER_MID,
            research_focus=str(d.get("research_focus") or ""),
            methodological_approach=list(d.get("methodological_approach") or []),
            communication_style=d.get("communication_style") or STYLE_TECHNICAL,
            citation_density=d.get("citation_density") or "moderate",
            explicit_consistency_constraints=str(
                d.get("explicit_consistency_constraints") or ""
            ),
        )

    @classmethod
    def from_categories(
        cls,
        categories: Iterable[str],
        *,
        career_stage: str = CAREER_MID,
        research_focus: str = "",
        methodological_approach: list[str] | None = None,
        communication_style: str = STYLE_TECHNICAL,
        citation_density: str = "moderate",
        explicit_consistency_constraints: str = "",
    ) -> "ScholarProfile":
        """
        从 Character.categories（旧类别标签）自动推导主/副领域与子领域。

        规则：
          - 第一个可解析的类别 → 主领域
          - 其余可解析的类别 → 副领域
          - 同一领域内多类别只保留首个为 subdomain
          - 全部无法解析时，默认人文学科
        """
        resolved: list[tuple[str, str]] = []
        seen_domains: set[str] = set()
        primary_domain: str = "humanities"
        subdomain: str = ""
        secondary: list[str] = []
        first = True

        for cat in categories:
            mapping = resolve_subdomain(cat)
            if mapping is None:
                continue
            dom, sub = mapping
            if first:
                primary_domain = dom
                subdomain = sub
                seen_domains.add(dom)
                first = False
                continue
            if dom in seen_domains:
                # 同领域内只把首个子领域保留为主 subdomain
                continue
            seen_domains.add(dom)
            secondary.append(dom)

        return cls(
            primary_domain=primary_domain,
            secondary_domains=secondary,
            subdomain=subdomain,
            career_stage=career_stage,
            research_focus=research_focus,
            methodological_approach=methodological_approach or [],
            communication_style=communication_style,
            citation_density=citation_density,
            explicit_consistency_constraints=explicit_consistency_constraints,
        )


def describe_profile(p: ScholarProfile) -> str:
    """人类可读的档案概要，用于 UI 或日志。"""
    primary = DOMAIN_LABELS.get(p.primary_domain, p.primary_domain)
    secondary = "、".join(
        DOMAIN_LABELS.get(d, d) for d in p.secondary_domains
    ) or "无"
    career = CAREER_LABELS.get(p.career_stage, p.career_stage)
    style = STYLE_LABELS.get(p.communication_style, p.communication_style)
    methods = "、".join(
        METHODOLOGICAL_APPROACHES.get(m, m) for m in p.methodological_approach
    ) or "默认"
    return (
        f"主领域：{primary}"
        + (f"（{p.subdomain}）" if p.subdomain else "")
        + f" | 副领域：{secondary} | 阶段：{career} | "
        f"沟通风格：{style} | 方法偏好：{methods}"
        + (f" | 研究焦点：{p.research_focus}" if p.research_focus else "")
    )
