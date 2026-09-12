TREATMENT_KEYWORDS = [
    "how should i treat",
    "how can i treat",
    "how do i treat",
    "how to treat",
    "treatment for",
    "treat my",
    "treat the",
    "what treatment",
    "which treatment",
]

PRESCRIPTION_KEYWORDS = [
    "prescribe a medication",
    "prescribe medication",
    "prescribe medicine",
    "prescribe a drug",
    "what medication should i take",
    "which medication should i take",
    "what medicine should i take",
    "which medicine should i take",
    "what drug should i take",
    "which drug should i take",
    "what dosage",
    "which dosage",
    "what dose",
    "which dose",
    "how much medication",
    "how much medicine",
]

HIGH_RISK_KEYWORDS = (
    TREATMENT_KEYWORDS
    + PRESCRIPTION_KEYWORDS
)


def validate_query(question: str) -> dict:
    normalized = question.lower().strip()

    for keyword in HIGH_RISK_KEYWORDS:
        if keyword in normalized:
            return {
                "allowed": False,
                "requires_human_validation": True,
                "reason": (
                    "Medical treatment or prescription-related "
                    "requests require human validation."
                ),
            }

    return {
        "allowed": True,
        "requires_human_validation": False,
        "reason": None,
    }
