"""
学者提示词验证标准。

提供两套检查：
  1. 静态检查（validate_profile / validate_enriched_prompt）
     在生成提示词前后做形式校验，确保 profile 与最终文本满足结构约束。
  2. 输出一致性检查（build_output_checklist）
     返回一份可注入到 System Prompt 末尾的「自检清单」，让 LLM 在生成回答前
     自检身份一致性、领域专业度、引用规范、学术严谨性。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from prompts.scholar_profile import (
    METHODOLOGICAL_APPROACHES,
    ScholarProfile,
)
from prompts.domain_templates import ALL_DOMAINS, DOMAIN_LABELS


@dataclass
class ValidationResult:
    ok: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


# ---------- 静态检查 ----------

def validate_profile(profile: ScholarProfile) -> ValidationResult:
    """对 ScholarProfile 做形式校验。"""
    res = ValidationResult(ok=True)

    # 1. 字段合法性（委托给 profile.validate）
    errs = profile.validate()
    res.errors.extend(errs)

    # 2. 副领域去重
    if len(set(profile.secondary_domains)) != len(profile.secondary_domains):
        res.warnings.append("secondary_domains 含重复项，已被自动去重处理")

    # 3. 方法论偏好与主领域一致性（启发式）
    primary = profile.primary_domain
    incompat: list[str] = []
    if primary == "natural_sciences":
        if "interpretive" in profile.methodological_approach:
            incompat.append("interpretive（诠释取向）较少用于自然科学主领域")
    if primary == "humanities":
        if "experimental" in profile.methodological_approach:
            incompat.append("experimental（实验取向）较少用于人文学科主领域")
    if primary == "engineering":
        if "critical" in profile.methodological_approach:
            incompat.append("critical（批判取向）较少用于工程学主领域")
    for msg in incompat:
        res.warnings.append(f"方法论偏好可能不匹配主领域：{msg}")

    # 4. 研究焦点长度
    if profile.research_focus and len(profile.research_focus) > 200:
        res.warnings.append(
            f"research_focus 过长（{len(profile.research_focus)} 字），"
            "建议精简到 200 字以内以保持提示词聚焦"
        )

    res.ok = not res.errors
    return res


def validate_enriched_prompt(text: str, profile: ScholarProfile) -> ValidationResult:
    """对 enrich_scholar_prompt 输出的文本做结构检查。"""
    res = validate_profile(profile)
    if not res.ok:
        return res

    # 必须包含的关键段标题（与 enricher 输出顺序对应）
    expected_markers = [
        "学者档案设定",
        f"主领域范式：{DOMAIN_LABELS.get(profile.primary_domain, profile.primary_domain)}",
        "职业阶段",
        "沟通风格",
        "引用密度",
    ]
    for m in expected_markers:
        if m not in text:
            res.errors.append(f"增强提示词缺少必要段落：{m}")

    # 跨领域学者必须含一致性原则
    if profile.secondary_domains and "跨领域一致性原则" not in text:
        res.errors.append("跨领域学者应包含「跨领域一致性原则」段落")

    res.ok = not res.errors
    return res


# ---------- 输出一致性检查（注入到 System Prompt 末尾） ----------

OUTPUT_CHECKLIST_TEMPLATE = """## 输出自检清单（生成回答前请默查）
在给出最终回答前，逐条对照本清单自检，不通过则修订后再输出：

### A. 身份一致性
1. 回答是否始终以 {character_name} 本人的立场、术语与价值取向发声？
   未出现「作为一名学者我认为」之类的身份标签废话。
2. 是否避免了用「该人物不会采用」的现代术语或立场？
3. 当人物跨多个领域时（{domain_summary}），主领域语调是否始终占主导，
   副领域仅作为参照视角出现，未发生风格漂移？

### B. 领域专业度
4. 是否使用了 {primary_domain_label} 的标准术语与方法论，
   而非泛化的「辩证地看」「从某种意义上说」这类空话？
5. 方法论偏好（{methods_summary}）是否在论证结构中体现？

### C. 引用规范
6. 重要论断是否给出可追溯依据（{citation_density_label}）？
7. 引用是否严格来自【参考资料】实际出现的段落或人物真实原作，
   而未虚构书名、页码或原文？

### D. 学术严谨
8. 是否区分了「事实」「主流理论」「争议假设」「本人推断」？
9. 是否在不确定处明说「现有材料不足以直接证明」「这超出我的直接经验」，
   而非含糊带过？
10. 推理过程是否完整可检：前提 → 证据 → 推导 → 结论 → 边界？
"""


def _methods_summary(methods: Iterable[str]) -> str:
    out = []
    for m in methods:
        if m in METHODOLOGICAL_APPROACHES:
            out.append(METHODOLOGICAL_APPROACHES[m].split("（")[0])
    return "、".join(out) if out else "默认"


def build_output_checklist(
    character_name: str, profile: ScholarProfile
) -> str:
    """生成可注入 System Prompt 的自检清单。"""
    primary_label = DOMAIN_LABELS.get(profile.primary_domain, profile.primary_domain)
    domain_summary = primary_label + (
        "、" + "、".join(
            DOMAIN_LABELS.get(d, d) for d in profile.secondary_domains
        ) if profile.secondary_domains else ""
    )
    density_label = {
        "sparse": "稀疏，仅关键论断处",
        "moderate": "适中，重要论断可追溯",
        "dense": "密集，每个非显然论断都应可追溯",
    }.get(profile.citation_density, "适中")

    return OUTPUT_CHECKLIST_TEMPLATE.format(
        character_name=character_name,
        domain_summary=domain_summary,
        primary_domain_label=primary_label,
        methods_summary=_methods_summary(profile.methodological_approach),
        citation_density_label=density_label,
    )


# ---------- 测试用例生成 ----------

DEFAULT_TEST_PROFILES: list[dict] = [
    # 单领域，纯哲学家，资深
    {
        "name": "晚期哲学家",
        "profile": ScholarProfile(
            primary_domain="humanities",
            subdomain="philosophy",
            career_stage="senior",
            communication_style="technical",
            citation_density="dense",
        ),
    },
    # 跨学科：经济学 + 政治学，中期，面向公众
    {
        "name": "政治经济学家（公众风格）",
        "profile": ScholarProfile(
            primary_domain="social_sciences",
            secondary_domains=["humanities"],
            subdomain="economics",
            career_stage="mid_career",
            communication_style="public_facing",
            citation_density="sparse",
        ),
    },
    # 自然科学 + 工程，早期，教学风格
    {
        "name": "工程物理学家（教学风格）",
        "profile": ScholarProfile(
            primary_domain="natural_sciences",
            secondary_domains=["engineering"],
            subdomain="physics",
            career_stage="early_career",
            communication_style="pedagogical",
            methodological_approach=["empirical", "experimental"],
            citation_density="moderate",
        ),
    },
    # 应用科学 + 人文，晚期，严谨风格
    {
        "name": "法哲学家",
        "profile": ScholarProfile(
            primary_domain="applied_sciences",
            secondary_domains=["humanities"],
            subdomain="law",
            career_stage="senior",
            communication_style="technical",
            methodological_approach=["interpretive", "critical"],
            citation_density="dense",
        ),
    },
]


def run_smoke_tests() -> list[dict]:
    """对 DEFAULT_TEST_PROFILES 跑一遍静态校验，返回结果列表（用于调试/CI）。"""
    from prompts.scholar_enricher import enrich_scholar_prompt

    results: list[dict] = []
    for case in DEFAULT_TEST_PROFILES:
        p: ScholarProfile = case["profile"]
        enriched = enrich_scholar_prompt(p)
        v = validate_enriched_prompt(enriched, p)
        results.append({
            "case": case["name"],
            "ok": v.ok,
            "errors": v.errors,
            "warnings": v.warnings,
            "enriched_length": len(enriched),
        })
    return results
