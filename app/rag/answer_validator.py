import re


MEDICAL_CATEGORIES = {
    "symptoms": [
        "symptom",
        "symptoms",
        "sign",
        "signs",
    ],
    "complications": [
        "complication",
        "complications",
    ],
    "risk": [
        "risk",
        "at risk",
        "risk factors",
    ],
    "causes": [
        "cause",
        "causes",
    ],
    "diagnosis": [
        "diagnosis",
        "diagnose",
        "diagnostic",
    ],
    "treatment": [
        "treatment",
        "treatments",
        "therapy",
        "therapies",
    ],
}


def normalize_text(text):
    text = str(text).lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def detect_answer_category(question):
    question = normalize_text(question)

    if any(
        keyword in question
        for keyword in MEDICAL_CATEGORIES["symptoms"]
    ):
        return "symptoms"

    if any(
        keyword in question
        for keyword in MEDICAL_CATEGORIES["risk"]
    ):
        return "risk"

    if any(
        keyword in question
        for keyword in MEDICAL_CATEGORIES["causes"]
    ):
        return "causes"

    if any(
        keyword in question
        for keyword in MEDICAL_CATEGORIES["diagnosis"]
    ):
        return "diagnosis"

    if any(
        keyword in question
        for keyword in MEDICAL_CATEGORIES["treatment"]
    ):
        return "treatment"

    return "general"


def validate_response(question, response, context):
    question_category = detect_answer_category(question)

    response_normalized = normalize_text(response)
    context_normalized = normalize_text(context)

    issues = []

    if not response_normalized:
        issues.append("empty_response")

    if question_category == "symptoms":

        complication_terms = [
            "aneurysm",
            "aneurysms",
            "chronic kidney disease",
            "heart attack",
            "heart failure",
            "stroke",
            "peripheral arterial disease",
        ]

        complication_matches = [
            term
            for term in complication_terms
            if term in response_normalized
        ]

        if complication_matches:
            issues.append(
                "possible_complications_presented_as_symptoms"
            )

    if response_normalized not in context_normalized:
        # This is informational only.
        # The generated answer is expected to synthesize
        # information rather than copy the source verbatim.
        pass

    return {
        "valid": len(issues) == 0,
        "question_category": question_category,
        "issues": issues,
    }


def build_safe_fallback(question, results):
    if not results:
        return (
            "The available medical information is insufficient "
            "to answer this question reliably."
        )

    best_document = results[0]["document"]

    answer = best_document.get("answer", "").strip()

    if not answer:
        return (
            "The available medical information is insufficient "
            "to answer this question reliably."
        )

    return (
        "According to the retrieved medical information:\n\n"
        + answer
        + "\n\n"
        "This information is provided for educational purposes "
        "and does not replace professional medical advice."
    )
