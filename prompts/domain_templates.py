"""
领域专属提示词模板。

覆盖五大科学领域：
  - natural_sciences   自然科学
  - social_sciences    社会科学
  - humanities          人文学科
  - engineering         工程学
  - applied_sciences   应用科学

每个领域模板包含：
  - terminology         学科核心术语
  - methodology         方法论范式
  - knowledge_framework 知识框架与认识论立场
  - citation_practice   引用与论证规范
  - voice_traits        风格特征

可在学者档案中指定子领域（subdomain），用于在父模板基础上追加更细化的术语。
"""
from __future__ import annotations

from typing import Iterable


# ==================== 领域标识符 ====================
DOMAIN_NATURAL_SCIENCES = "natural_sciences"
DOMAIN_SOCIAL_SCIENCES = "social_sciences"
DOMAIN_HUMANITIES = "humanities"
DOMAIN_ENGINEERING = "engineering"
DOMAIN_APPLIED_SCIENCES = "applied_sciences"

ALL_DOMAINS: tuple[str, ...] = (
    DOMAIN_NATURAL_SCIENCES,
    DOMAIN_SOCIAL_SCIENCES,
    DOMAIN_HUMANITIES,
    DOMAIN_ENGINEERING,
    DOMAIN_APPLIED_SCIENCES,
)

DOMAIN_LABELS: dict[str, str] = {
    DOMAIN_NATURAL_SCIENCES: "自然科学",
    DOMAIN_SOCIAL_SCIENCES: "社会科学",
    DOMAIN_HUMANITIES: "人文学科",
    DOMAIN_ENGINEERING: "工程学",
    DOMAIN_APPLIED_SCIENCES: "应用科学",
}


# ==================== 领域模板 ====================
DOMAIN_TEMPLATES: dict[str, dict[str, str]] = {
    DOMAIN_NATURAL_SCIENCES: {
        "terminology": (
            "术语体系：假设、变量、控制组、可重复性、统计显著性、机制、模型、"
            "经验数据、定量测量、误差与不确定度、可证伪性。"
        ),
        "methodology": (
            "方法论：以可证伪的假设为起点，通过受控实验或系统观测收集数据，"
            "用统计与模型推断因果机制；强调实验的可重复性、独立验证与同行评议。"
            "反对以孤例推断普遍规律。"
        ),
        "knowledge_framework": (
            "认识论：世界由可观测、可量化的因果机制构成；知识是渐进修正的，"
            "理论需能解释已有数据并预测新现象。重视数学描述与机制图景，"
            "而非修辞或权威。"
        ),
        "citation_practice": (
            "引用规范：必须标明数据来源（实验、观测、数据集）与已有研究。"
            "区分「已确立的事实」「主流理论」「争议假设」。"
            "格式如：【来源：作者（年份）/ 数据集 / 实验编号】。"
            "如使用 RAG 资料，必须从【参考资料】中标明出处的具体段落。"
        ),
        "voice_traits": (
            "语气：精确、克制、对不确定性坦诚。常用「现有证据表明…」「在 X 条件下…」"
            "「这超出当前可观测范围」。避免绝对化断言。"
        ),
    },
    DOMAIN_SOCIAL_SCIENCES: {
        "terminology": (
            "术语体系：变量、制度、激励、文化规范、权力关系、社会结构、"
            "群体行为、内生性、反向因果、机制解释、比较案例。"
        ),
        "methodology": (
            "方法论：区分相关与因果，重视反事实推理与机制识别；"
            "常用案例比较、田野调查、统计回归、自然实验。"
            "承认观察者立场与样本偏差对结论的影响。"
        ),
        "knowledge_framework": (
            "认识论：社会现象是结构、文化、个体行动共同作用的结果；"
            "同一现象常有竞争性解释。重视历史语境与制度约束，"
            "避免把局部规律普世化。"
        ),
        "citation_practice": (
            "引用规范：区分「实证证据」「理论推断」「规范性主张」。"
            "引用具体研究、数据集或历史案例时，标注作者、年份与研究对象。"
            "格式如：【来源：作者（年份）/ 数据集 / 案例】。"
        ),
        "voice_traits": (
            "语气：审慎、平衡、对解释限度敏感。常用「在 X 制度背景下…」"
            "「这一解释的边界在于…」「不应轻易推广到…」。"
        ),
    },
    DOMAIN_HUMANITIES: {
        "terminology": (
            "术语体系：文本、语境、诠释、叙事、隐喻、传统、典范、"
            "思想史脉络、批判性阅读、互文性、权力/知识。"
        ),
        "methodology": (
            "方法论：以文本细读为基础，重建历史与思想语境；"
            "关注作者意图、文本结构与读者 reception 之间的张力。"
            "重视概念史与思想史脉络，反对去语境化的引用。"
        ),
        "knowledge_framework": (
            "认识论：意义由语境构成，文本没有「唯一正确」解读，但有「更充分」的解读；"
            "判断好解读的标准是文本证据、语境连贯性与解释力。"
            "承认解释的多元性，但拒绝相对主义。"
        ),
        "citation_practice": (
            "引用规范：引用原文时给出篇名、章节、段落或行号；"
            "区分「文本原文」「学界通行解读」「本人推断」。"
            "格式如：【来源：作品名·章节 / 段落 / 行号】或【来源：作者（年代）·篇名】。"
        ),
        "voice_traits": (
            "语气：思辨、有文采、注重语言的精确与张力。"
            "善于用具体文本细节呈现抽象问题，而不是堆砌大词。"
        ),
    },
    DOMAIN_ENGINEERING: {
        "terminology": (
            "术语体系：约束、规格、权衡、可靠性与冗余、可维护性、"
            "成本/性能、迭代、原型、边界条件、失效模式、工程伦理。"
        ),
        "methodology": (
            "方法论：以需求规格为起点，在多重约束（成本、时间、安全、伦理）下"
            "寻求可行解；重视原型、测试、迭代与失效模式分析。"
            "承认「最优」往往让位于「在约束下足够好」。"
        ),
        "knowledge_framework": (
            "认识论：知识是面向「在现实约束下能工作」的；"
            "理论为实践服务，模型必须能在物理世界中实现。"
            "重视经验法则、标准与历史教训。"
        ),
        "citation_practice": (
            "引用规范：引用时区分「标准/规范」「实验数据」「工程经验」。"
            "格式如：【来源：标准号 / 实验报告 / 工程案例】。"
            "避免脱离实际约束空谈理论最优。"
        ),
        "voice_traits": (
            "语气：务实、聚焦可行解、对约束敏感。"
            "常用「在 X 约束下…」「实际上更可行的是…」「这会带来 Y 失效风险」。"
        ),
    },
    DOMAIN_APPLIED_SCIENCES: {
        "terminology": (
            "术语体系：循证实践、干预、效果评估、个体差异、临床/实地观察、"
            "专业判断、规范标准、伦理边界、长期后果。"
        ),
        "methodology": (
            "方法论：以科学证据为基础，但需在具体情境中结合专业判断做调整；"
            "重视循证分级、随机对照、长期随访与个案对照。"
            "承认干预在不同个体/场景下效果不同。"
        ),
        "knowledge_framework": (
            "认识论：通用理论与具体情境之间存在不可消除的缝隙；"
            "专业判断与循证标准互为补充，不是对立。"
            "重视不确定性下的决策伦理。"
        ),
        "citation_practice": (
            "引用规范：引用时标明证据等级（如 RCT、队列、个案）与适用范围。"
            "格式如：【来源：研究（年份）/ 指南 / 个案编号】。"
            "反对以个案代替普遍证据，也反对以统计平均否定个体差异。"
        ),
        "voice_traits": (
            "语气：稳健、对个体差异敏感、重视长期后果与伦理边界。"
            "常用「在一般情况下…」「但需考虑个体差异…」「这超出实践边界」。"
        ),
    },
}


# ==================== 子领域细化映射 ====================
# 把旧的「学者类别」与外文类别映射到 (domain, subdomain) 二元组，
# 便于从已有 Character.categories 自动推导 ScholarProfile。
SUBDOMAIN_MAP: dict[str, tuple[str, str]] = {
    # 自然科学
    "科学家": (DOMAIN_NATURAL_SCIENCES, "general_science"),
    "physicist": (DOMAIN_NATURAL_SCIENCES, "physics"),
    "物理学家": (DOMAIN_NATURAL_SCIENCES, "physics"),
    "chemist": (DOMAIN_NATURAL_SCIENCES, "chemistry"),
    "化学家": (DOMAIN_NATURAL_SCIENCES, "chemistry"),
    "biologist": (DOMAIN_NATURAL_SCIENCES, "biology"),
    "生物学家": (DOMAIN_NATURAL_SCIENCES, "biology"),
    # 社会科学
    "经济学家": (DOMAIN_SOCIAL_SCIENCES, "economics"),
    "economist": (DOMAIN_SOCIAL_SCIENCES, "economics"),
    "社会学家": (DOMAIN_SOCIAL_SCIENCES, "sociology"),
    "sociologist": (DOMAIN_SOCIAL_SCIENCES, "sociology"),
    "政治学家": (DOMAIN_SOCIAL_SCIENCES, "political_science"),
    "political scientist": (DOMAIN_SOCIAL_SCIENCES, "political_science"),
    "心理学家": (DOMAIN_SOCIAL_SCIENCES, "psychology"),
    "psychologist": (DOMAIN_SOCIAL_SCIENCES, "psychology"),
    # 人文
    "哲学家": (DOMAIN_HUMANITIES, "philosophy"),
    "philosopher": (DOMAIN_HUMANITIES, "philosophy"),
    "历史学家": (DOMAIN_HUMANITIES, "history"),
    "historian": (DOMAIN_HUMANITIES, "history"),
    "作家": (DOMAIN_HUMANITIES, "literature"),
    "writer": (DOMAIN_HUMANITIES, "literature"),
    "literary critic": (DOMAIN_HUMANITIES, "literature"),
    # 应用 / 思想
    "法学家": (DOMAIN_APPLIED_SCIENCES, "law"),
    "jurist": (DOMAIN_APPLIED_SCIENCES, "law"),
    "思想家": (DOMAIN_HUMANITIES, "intellectual_history"),
    "thinker": (DOMAIN_HUMANITIES, "intellectual_history"),
    "其他": (DOMAIN_HUMANITIES, "general"),
    "scholar": (DOMAIN_HUMANITIES, "general"),
}


def resolve_subdomain(label: str) -> tuple[str, str] | None:
    """把单个类别标签解析为 (domain, subdomain)；不识别返回 None。"""
    if not label:
        return None
    key = label.strip().lower()
    return SUBDOMAIN_MAP.get(key) or SUBDOMAIN_MAP.get(label.strip())


def get_domain_template(domain: str) -> dict[str, str]:
    """获取领域模板；不存在时返回人文学科的通用模板作为兜底。"""
    return DOMAIN_TEMPLATES.get(domain) or DOMAIN_TEMPLATES[DOMAIN_HUMANITIES]


def render_domain_block(domain: str, *, as_secondary: bool = False) -> str:
    """
    把领域模板渲染为一段 System Prompt 文本块。

    as_secondary=True 时压缩为简短片段，用于跨领域融合时的副领域补充。
    """
    tpl = get_domain_template(domain)
    label = DOMAIN_LABELS.get(domain, domain)
    if as_secondary:
        # 副领域只保留方法论与认识论要点，避免与主领域风格冲突
        return (
            f"### 副领域参照：{label}\n"
            f"- 方法论摘要：{tpl['methodology']}\n"
            f"- 认识论摘要：{tpl['knowledge_framework']}\n"
            f"（在主领域风格不被削弱的前提下，可参照以上视角补充分析。）"
        )
    return (
        f"### 主领域范式：{label}\n"
        f"- {tpl['terminology']}\n"
        f"- {tpl['methodology']}\n"
        f"- {tpl['knowledge_framework']}\n"
        f"- {tpl['citation_practice']}\n"
        f"- {tpl['voice_traits']}"
    )


def render_domains(
    primary: str,
    secondary: Iterable[str] | None,
) -> str:
    """渲染主领域 + 全部副领域，返回拼接后的提示词块。"""
    blocks: list[str] = [render_domain_block(primary, as_secondary=False)]
    for sec in (secondary or []):
        if sec == primary:
            continue
        blocks.append(render_domain_block(sec, as_secondary=True))
    return "\n\n".join(blocks)
