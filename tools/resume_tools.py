from agents.career_agent import analyze_resume_text


def analyze_resume(resume_text: str) -> str:
    result = analyze_resume_text(resume_text)
    skills = result["skills"]

    if not skills:
        return "没有识别到明确技能关键词。建议补充项目技术栈、使用场景和项目结果。"

    suggestions = "\n".join(
        f"{index}. {suggestion}"
        for index, suggestion in enumerate(result["suggestions"], start=1)
    )

    return f"""
简历技能分析结果：

识别到的技能：
{", ".join(skills)}

建议：
{suggestions}
"""
