from agents.career_agent import analyze_jd_text


def analyze_jd(jd_text: str) -> str:
    result = analyze_jd_text(jd_text)
    keywords = result["keywords"]

    if not keywords:
        return """
JD 分析结果：

没有识别到明确技术关键词。

建议：
1. 检查 JD 原文是否过短
2. 尝试提供更完整的岗位描述
"""

    focus = "\n".join(
        f"{index}. {item}"
        for index, item in enumerate(result["preparation_focus"], start=1)
    )

    return f"""
JD 分析结果：

识别到的关键词：
{", ".join(keywords)}

面试准备重点：
{focus}
"""
