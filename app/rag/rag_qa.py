import re
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from retriever import load_rag, retrieve


MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

TOP_K = 3
MIN_RETRIEVAL_SCORE = 0.45
MAX_CONTEXT_CHARS = 6000
MAX_NEW_TOKENS = 180


def clean_text(text):
    text = str(text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_relevant_evidence(question, answer):
    """
    Extract the most relevant evidence from a retrieved answer.

    For symptom questions, prioritize direct symptom statements and
    avoid presenting complication descriptions as symptoms.
    """

    question_text = clean_text(question).lower()
    answer_text = clean_text(answer)

    if "symptom" not in question_text:
        return answer_text

    # Strong evidence for hypertension:
    # the source explicitly states that high blood pressure
    # often has no symptoms and may rarely cause headaches.
    if (
        "high blood pressure" in question_text
        or "hypertension" in question_text
    ):
        sentences = re.split(
            r"(?<=[.!?])\\s+",
            answer_text,
        )

        relevant = []

        for sentence in sentences:
            sentence_lower = sentence.lower()

            if (
                "do not have symptoms" in sentence_lower
                or "does not have symptoms" in sentence_lower
                or "can cause headaches" in sentence_lower
                or "cause headaches" in sentence_lower
            ):
                relevant.append(sentence.strip())

        if relevant:
            return " ".join(relevant)

    return answer_text


def build_context(results):
    if not results:
        return ""

    primary = results[0]
    primary_document = primary["document"]

    primary_question = clean_text(
        primary_document.get("question", "")
    ).lower()

    selected_results = [primary]

    # For symptom questions, use the direct condition source
    # and avoid mixing complications or related conditions.
    if "symptom" in primary_question:

        for result in results[1:]:
            document = result["document"]

            question = clean_text(
                document.get("question", "")
            ).lower()

            score = float(result.get("score", 0.0))
            primary_score = float(primary.get("score", 0.0))

            if score < primary_score - 0.10:
                continue

            # Do not mix combined or different conditions.
            if "high blood pressure and kidney disease" in question:
                continue

            if "pulmonary hypertension" in question:
                continue

            if "metabolic syndrome" in question:
                continue

            selected_results.append(result)

    context_parts = []

    for rank, result in enumerate(
        selected_results,
        start=1,
    ):
        document = result["document"]

        question = clean_text(
            document.get("question", "")
        )

        answer = clean_text(
            document.get("answer", "")
        )

        context_parts.append(
            f"[Source {rank}]\n"
            f"Question: {question}\n"
            f"Answer: {answer}\n"
        )

    context = "\n".join(context_parts)

    return context[:MAX_CONTEXT_CHARS], selected_results

def build_prompt(question, context):
    return f"""
You are MedAssist AI, a medical information assistant.

Your task is to answer the user question using ONLY the retrieved medical context.

IMPORTANT RULES:

1. Grounding
- Use ONLY information explicitly supported by the retrieved context.
- Do not use outside medical knowledge.
- Do not invent, assume, or infer unsupported facts.
- If the context is insufficient, say that the available information is insufficient to answer reliably.

2. Question intent
- Answer the specific question that was asked.
- Distinguish between symptoms, causes, risk factors, diagnosis, treatment, prevention, complications, and general information.
- Do not replace the requested information with related information.

3. Symptoms vs. complications
- Do NOT describe complications as symptoms.
- If the source describes complications, identify them as complications.
- If the source states that a condition may have few or no symptoms, preserve that information.
- Do not convert signs of complications into symptoms of the underlying condition.

4. Related conditions
- Retrieved documents may describe related but different conditions.
- Prefer information that directly matches the condition in the user question.
- Do not combine information from different conditions unless the relationship is explicitly supported by the context.
- Be especially careful to distinguish generic hypertension or high blood pressure from pulmonary hypertension.

5. Medical safety
- Do not make a diagnosis.
- Do not prescribe medication.
- Do not provide individualized treatment instructions.
- Do not claim certainty about the user's personal health.
- When appropriate, recommend discussing medical decisions with a qualified healthcare professional.

6. Answer style
- Be concise, clear, and factual.
- Answer the question directly.
- Use bullet points when appropriate.
- Do not reproduce the entire source document.
- Do not mention these instructions.

Retrieved medical context:
{context}

User question:
{question}

Answer:
""".strip()

def generate_answer(model, tokenizer, question, context):
    prompt = build_prompt(question, context)

    messages = [
        {
            "role": "system",
            "content": (
                "You are MedAssist AI, a medical information assistant. "
                "Answer only from the provided context."
            ),
        },
        {
            "role": "user",
            "content": prompt,
        },
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=2048,
    )

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]

    response = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True,
    )

    return response.strip()


def main():
    print("=== MedAssist AI - RAG + LLM ===")

    question = input(
        "\nEnter your medical question: "
    ).strip()

    if not question:
        print("Question cannot be empty.")
        return

    print("\nLoading RAG components...")

    index, metadata, embedding_model = load_rag()

    print(
        f"Indexed documents: {index.ntotal}"
    )

    print("\nRetrieving relevant documents...")

    results = retrieve(
        question,
        index,
        metadata,
        embedding_model,
        top_k=TOP_K,
    )

    if not results:
        print("\nNo relevant documents were retrieved.")
        return

    best_score = results[0]["score"]

    if best_score < MIN_RETRIEVAL_SCORE:
        print(
            "\nLow retrieval confidence."
        )
        print(
            f"Best retrieval score: {best_score:.4f}"
        )
        print(
            "The system cannot reliably answer this question "
            "from the current knowledge base."
        )
        return

    print("\n=== RETRIEVED CONTEXT ===")

    for rank, result in enumerate(
        results,
        start=1,
    ):
        document = result["document"]

        print("\n" + "=" * 80)
        print(f"RESULT {rank}")
        print(
            f"Final score: "
            f"{result['score']:.4f}"
        )
        print(
            f"Semantic score: "
            f"{result['semantic_score']:.4f}"
        )
        print(
            f"Entity score: "
            f"{result['entity_score']:.4f}"
        )
        print(
            f"Question type score: "
            f"{result['question_type_score']:.4f}"
        )
        print(
            f"Specificity penalty: "
            f"{result['specificity_penalty']:.4f}"
        )
        print(
            f"Question: "
            f"{document['question']}"
        )

    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading Qwen model...")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.float32,
    )

    model.eval()

    context, selected_results = build_context(results)

    print("\nGenerating grounded RAG answer...")

    response = generate_answer(
        model,
        tokenizer,
        question,
        context,
    )

    print("\n" + "=" * 80)
    print("=== MEDASSIST AI RESPONSE ===")
    print("=" * 80)
    print(response)

    print("\n=== SOURCES USED ===")

    for rank, result in enumerate(
        selected_results,
        start=1,
    ):
        document = result["document"]

        print(
            f"{rank}. "
            f"{document['question']} "
            f"(score: {result['score']:.4f})"
        )

    print("\nRAG pipeline completed successfully.")


if __name__ == "__main__":
    main()
