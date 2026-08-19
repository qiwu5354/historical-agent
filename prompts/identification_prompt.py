"""
人物识别提示词：根据用户输入的名字，识别出标准人物身份、所属学者类别，
并生成后续资料检索所需的查询词。
"""
from __future__ import annotations

from prompts import safe_format


IDENTIFY_SYSTEM = """你是一位严谨的人物信息识别专家。你的任务是根据用户输入的名字，
结合可用的网络搜索片段，判断这个“名字”最可能指向哪位真实历史人物/学者。

要求：
1. 如果输入是别名、简称、外文名，请还原为最常用的标准全名。
2. 判断该人物所属的学者/思想类别，一个人物可以同时属于多个类别。
3. 不要编造；如果无法确定具体人物，name 仍使用输入名，并在 summary 中说明“可能指代不明”。
4. 为后续检索生成高质量搜索词，覆盖三类资料：
   - 人物自己的作品/著作/自传
   - 他人写的传记/评传/研究
   - 他所在历史阶段的权威史料/历史背景
5. 搜索词应中英文混合，具体、可执行，不要泛泛的“人物简介”。
6. 同名消歧：如果用户输入的名字可能对应多个不同的真实人物（例如“马克思”可能是
   卡尔·马克思也可能是卡尔·马克思的兄弟、或同名的其他人物；“林肯”可能指总统
   亚伯拉罕·林肯也可能指其他同名人物），必须在 candidates 中列出所有合理候选。
   只有当名字明确指向唯一人物时才 ambiguous=false。

可用类别（不限于）：
- 哲学家
- 心理学家
- 经济学家
- 社会学家
- 历史学家
- 政治学家
- 法学家
- 作家
- 科学家
- 思想家
- 其他

输出严格 JSON。"""

IDENTIFY_USER = """用户输入的人物名：{input_name}

【初步网络检索片段】
{snippets}

请根据以上信息，输出如下 JSON 结构：

{
  "name": "最可能的标准全名",
  "aliases": ["常用别名", "外文名"],
  "categories": ["哲学家", "经济学家"],
  "era": "生卒年，如 1723-1790",
  "nation": "国家/地区",
  "summary": "一句话说明此人是谁",
  "works": ["《国富论》", "《道德情操论》"],
  "work_queries": ["亚当·斯密 国富论 全文", "Adam Smith Wealth of Nations full text"],
  "biography_queries": ["亚当·斯密 传记", "Adam Smith biography"],
  "history_queries": ["亚当·斯密 启蒙时代 历史背景", "18世纪欧洲 思想史 档案"],
  "ambiguous": false,
  "candidates": []
}

同名消歧时的输出（ambiguous=true 时）：
{
  "name": "默认选中的候选（通常是知名度最高的一位）",
  "ambiguous": true,
  "candidates": [
    {
      "name": "候选1标准全名",
      "era": "生卒年",
      "nation": "国家/地区",
      "categories": ["哲学家"],
      "summary": "一句话区分此人（例如：德国哲学家，马克思主义创始人）"
    },
    {
      "name": "候选2标准全名",
      "era": "生卒年",
      "nation": "国家/地区",
      "categories": ["经济学家"],
      "summary": "一句话区分此人"
    }
  ]
}

注意：
- 只有当输入名确实可能指向多个真实人物时才输出 candidates（2~4 个为宜）。
- ambiguous=false 时 candidates 必须为空数组。
- categories 必须从可用类别中选择，可以是多个。
- work_queries 用于搜索他本人的著作/文章。
- biography_queries 用于搜索别人写的传记、评传、回忆录。
- history_queries 用于搜索他所在历史时期的官方史料、权威历史背景。
- 如果无法从检索片段确认，name 可保留用户输入，categories 给 ["其他"]，但搜索词仍要尽量合理。"""

IDENTIFY_USER_HINT = """
【用户已确认目标人物】
用户已明确选择目标人物：{hint}
请直接以该人物为准输出完整档案（含检索词），不要输出候选列表，ambiguous 固定为 false。
"""


def build_identification_messages(
    input_name: str, snippets_text: str, hint: str | None = None
) -> list[dict[str, str]]:
    user_content = safe_format(
        IDENTIFY_USER,
        input_name=input_name,
        snippets=snippets_text or "（暂无检索结果，请基于你的知识判断）",
    )
    if hint:
        user_content += "\n\n" + safe_format(IDENTIFY_USER_HINT, hint=hint)
    return [
        {"role": "system", "content": IDENTIFY_SYSTEM},
        {"role": "user", "content": user_content},
    ]
