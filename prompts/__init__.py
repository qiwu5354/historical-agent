"""
Prompts package: shared utilities for template formatting.
"""


def safe_format(template: str, **kwargs: str) -> str:
    """
    安全模板替换：用 str.replace 逐 key 替换，避免资料中的花括号
    被 Python str.format() 误解析导致 KeyError。
    """
    result = template
    for key, value in kwargs.items():
        result = result.replace("{" + key + "}", value)
    return result
