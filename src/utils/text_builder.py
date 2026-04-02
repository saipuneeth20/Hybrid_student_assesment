def build_input_text(
    question: str,
    reference_answer: str,
    student_answer: str,
    context: str | None = None
) -> str:
    """
    Construct input text for the student scoring model.

    Format (with context):
        context: ...
        question: ...
        reference: ...
        student: ...

    Format (without context):
        question: ...
        reference: ...
        student: ...
    """
    question = question.strip() if isinstance(question, str) else ""
    reference_answer = reference_answer.strip() if isinstance(reference_answer, str) else ""
    student_answer = student_answer.strip() if isinstance(student_answer, str) else ""
    context = context.strip() if isinstance(context, str) else ""

    if context:
        text = (
            f"context: {context}\n"
            f"question: {question}\n"
            f"reference: {reference_answer}\n"
            f"student: {student_answer}"
        )
    else:
        text = (
            f"question: {question}\n"
            f"reference: {reference_answer}\n"
            f"student: {student_answer}"
        )

    return text