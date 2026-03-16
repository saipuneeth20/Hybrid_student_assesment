# src/utils/text_builder.py

def build_input_text(
    question: str,
    answer: str,
    context: str | None = None
) -> str:
    """
    Construct input text for the student scoring model.

    Format (with context):
        context: ...
        question: ...
        answer: ...

    Format (without context):
        question: ...
        answer: ...

    This function MUST be used consistently across
    training, validation, and inference.
    """

    question = question.strip() if isinstance(question, str) else ""
    answer = answer.strip() if isinstance(answer, str) else ""
    context = context.strip() if isinstance(context, str) else ""

    if context != "":
        text = (
            f"context: {context}\n"
            f"question: {question}\n"
            f"answer: {answer}"
        )
    else:
        text = (
            f"question: {question}\n"
            f"answer: {answer}"
        )

    return text
