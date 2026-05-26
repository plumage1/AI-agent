from rag.rag_chain import answer_with_rag_and_sources


def format_source(source: dict, index: int) -> str:
    return (
        f"来源 {index}：{source['source_file']} / {source['title']}\n"
        f"相关分数：{source.get('score', '-')}\n"
        f"{source['content']}"
    )


def rag_search(query: str) -> str:
    result = answer_with_rag_and_sources(query)

    if not result["sources"]:
        return result["answer"]

    sources_text = "\n\n".join(
        format_source(source, index + 1)
        for index, source in enumerate(result["sources"])
    )

    return f"""
{result["answer"]}

参考来源：
{sources_text}
"""
