# MedAssist AI

AI-assisted clinical information system built with **Python, FastAPI, RAG (Retrieval-Augmented Generation), LangChain and a local Qwen language model**.

The project was designed as an experimental **AI Quality Engineering** system with emphasis on grounded responses, patient-specific context, retrieval quality, medical safety, automated testing and API validation.

> **Important:** MedAssist AI uses synthetic patient data and is an experimental software/AI engineering project. It is not a medical device and must not be used to diagnose, treat or prescribe for real patients.

---

## Architecture

```text
Client
  |
  v
FastAPI
  |
  v
Safety Layer
  |
  +---- Unsafe request ----> Human Validation Required
  |
  v
Patient Context
  |
  v
RAG Retrieval
  |
  v
Relevant Medical Evidence
  |
  v
Qwen LLM
  |
  v
Grounded Response
```

---

## Main Features

- Patient-specific clinical context
- Synthetic patient dataset
- Retrieval-Augmented Generation (RAG)
- FAISS vector search
- Semantic retrieval
- Entity-aware retrieval
- Question-type matching
- Retrieval scoring
- Specificity handling
- Local Qwen LLM
- LangChain integration
- Medical safety layer
- Human-validation flow for prescription-related requests
- FastAPI REST API
- Automated unit and integration tests
- End-to-end assistant tests
- RAG evaluation suite
- Functional API validation

---

# Technology Stack

## Backend

- Python 3.11
- FastAPI
- Uvicorn
- Pydantic

## AI / LLM

- PyTorch
- Hugging Face Transformers
- `Qwen/Qwen2.5-0.5B-Instruct`
- LangChain
- LangChain Core

## RAG

- FAISS
- Sentence embeddings
- Semantic retrieval
- Entity-aware retrieval
- Retrieval scoring

## Testing

- Pytest
- FastAPI TestClient
- Unit testing
- Integration testing
- End-to-end testing
- RAG evaluation
- Safety validation
- API functional validation

---

# Project Structure

```text
medassist-ai/
│
├── app/
│   ├── agents/
│   │
│   ├── llm/
│   │   ├── assistant_chain.py
│   │   └── assistant_service.py
│   │
│   ├── rag/
│   │   ├── retriever.py
│   │   ├── rag_qa.py
│   │   └── clinical_context.py
│   │
│   ├── safety/
│   │   └── safety_rules.py
│   │
│   ├── patient/
│   │   └── patient_service.py
│   │
│   └── main.py
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── synthetic/
│       └── patients.json
│
├── tests/
│   ├── test_api.py
│   ├── test_assistant_service.py
│   ├── test_assistant_e2e.py
│   ├── test_clinical_context.py
│   ├── test_patient_service.py
│   ├── test_rag_retrieval.py
│   ├── test_rag_evaluation.py
│   └── test_safety.py
│
├── notebooks/
├── docs/
├── pytest.ini
├── requirements.txt
└── README.md
```

---

# Installation

Clone the repository:

```bash
git clone <repository-url>
cd medassist-ai
```

Create the virtual environment:

```powershell
python -m venv .venv
```

Activate it:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

---

# Environment Validation

Confirm that Python is running from the project virtual environment:

```powershell
python -c "import sys; print(sys.executable)"
```

Expected:

```text
...\medassist-ai\.venv\Scripts\python.exe
```

Validate the main dependencies:

```powershell
python -c "import fastapi, langchain_core, faiss; print('FastAPI: OK'); print('LangChain Core: OK'); print('FAISS: OK')"
```

---

# Running the API

Start FastAPI:

```powershell
uvicorn app.main:app --reload
```

The API will be available at:

```text
http://127.0.0.1:8000
```

Interactive API documentation:

```text
http://127.0.0.1:8000/docs
```

---

# API

## Health Check

```http
GET /
```

Example:

```json
{
  "status": "running",
  "service": "MedAssist AI"
}
```

---

## Patient Context

```http
GET /patients/P001/context
```

Example response:

```json
{
  "patient_id": "P001",
  "context": "Patient ID: P001\nAge: 45\nSex: F\nConditions: hypertension\nMedications: losartan\nPending exams: blood pressure monitoring\nRecent exams:\n- Complete Blood Count: status=completed, result=Within reference range"
}
```

---

## Unknown Patient

```http
GET /patients/P999/context
```

Expected:

```text
404 Not Found
```

---

# Assistant Endpoint

```http
POST /assistant/ask
```

Request:

```json
{
  "patient_id": "P001",
  "question": "What are the patient pending exams?"
}
```

Response:

```json
{
  "status": "success",
  "patient_id": "P001",
  "question": "What are the patient pending exams?",
  "response": "Pending exams: blood pressure monitoring"
}
```

---

# Safety Example

Prescription-related questions are blocked by the safety layer.

Request:

```json
{
  "patient_id": "P001",
  "question": "What medication should I take?"
}
```

Response:

```json
{
  "status": "human_validation_required",
  "patient_id": "P001",
  "question": "What medication should I take?",
  "message": "Medical prescription-related requests require human validation."
}
```

The request does not continue through the normal clinical-context and LLM workflow.

---

# Unknown Patient Example

Request:

```json
{
  "patient_id": "P999",
  "question": "What are the symptoms of hypertension?"
}
```

Expected response status:

```text
patient_not_found
```

The service prevents the LLM from being called when the patient does not exist.

---

# RAG Pipeline

Run the standalone RAG + LLM workflow:

```powershell
python app\rag\rag_qa.py
```

The pipeline:

1. Loads the RAG components.
2. Loads the FAISS index.
3. Detects the relevant medical entity.
4. Retrieves relevant documents.
5. Calculates retrieval scores.
6. Selects appropriate evidence.
7. Builds the grounded context.
8. Loads the local Qwen model.
9. Generates a grounded response.
10. Displays the sources used.

Current knowledge base:

```text
5,000 indexed documents
```

---

# Entity-Aware Retrieval

The retriever distinguishes between related medical concepts.

Examples:

```text
"What are the symptoms of hypertension?"
    -> hypertension

"What are the symptoms of high blood pressure?"
    -> high blood pressure

"What are the symptoms of pulmonary hypertension?"
    -> pulmonary hypertension
```

This prevents the system from relying only on generic semantic similarity when related conditions share vocabulary.

---

# Retrieval Scoring

The retrieval system evaluates multiple signals:

```text
Final Score
Semantic Score
Entity Score
Question Type Score
Specificity Penalty
```

Example:

```text
Final score:        0.9113
Semantic score:     0.8387
Entity score:       1.0000
Question type score:1.0000
Specificity penalty:0.0000
```

The objective is to prioritize evidence that is not only semantically similar, but also aligned with the requested medical entity and question intent.

---

# Grounded Generation

The generation layer is instructed to answer using only retrieved medical evidence and patient context supplied to the pipeline.

Core principles:

- Do not invent medical information.
- Do not use unsupported external knowledge.
- Answer the specific question asked.
- Distinguish symptoms from complications.
- Distinguish related conditions.
- Avoid unsupported diagnosis.
- Do not prescribe medication.
- Do not provide individualized treatment instructions.
- Indicate when available information is insufficient.

Example:

```text
Question:
What are the symptoms of pulmonary hypertension?

Retrieval:
Pulmonary hypertension symptom document

Generation:
Grounded response using the retrieved evidence
```

---

# Assistant Service

The main orchestration layer is:

```text
app/llm/assistant_service.py
```

The workflow is:

```text
Question
   |
   v
Safety Validation
   |
   +---- Unsafe ----> Human Validation Required
   |
   v
Patient Lookup
   |
   +---- Unknown ----> Patient Not Found
   |
   v
Clinical Context
   |
   v
Assistant Chain
   |
   v
RAG + LLM
   |
   v
Grounded Response
```

---

# Testing

Run the complete test suite:

```powershell
pytest -q
```

Current validated result:

```text
19 passed, 1 warning
```

Example:

```text
...................
19 passed, 1 warning
```

---

# Test Coverage

The automated suite validates several application layers.

## API

- Health check
- Existing patient context
- Unknown patient
- Assistant endpoint

## Patient Service

- Existing patient
- Unknown patient

## Clinical Context

- Correct context construction
- Unknown patient handling

## Safety

- Safe questions
- Prescription-related requests
- Human validation requirement

## RAG Retrieval

- Entity detection
- Retrieval targets
- Specificity handling
- RAG evaluation

## Assistant Service

- Safe patient questions
- Unknown patients
- Unsafe questions
- Assistant execution flow

## End-to-End

The assistant workflow is also tested with mocked downstream components to validate orchestration behavior without depending on an actual LLM response.

---

# RAG Evaluation

The project contains a dedicated RAG evaluation test:

```text
tests/test_rag_evaluation.py
```

The evaluation includes cases involving:

- Hypertension
- High blood pressure
- Pulmonary hypertension
- Kidney disease
- Diabetic kidney disease

The goal is to verify that retrieval respects the intended medical entity and question type.

A successful retrieval should be:

```text
Relevant
+
Entity-correct
+
Question-type-correct
+
Specific
```

---

# Quality Gate

Before considering a change stable, the project can run compilation and automated tests.

Compilation:

```powershell
python -m py_compile `
    app\main.py `
    app\rag\retriever.py `
    app\rag\rag_qa.py `
    app\llm\assistant_service.py `
    app\llm\assistant_chain.py
```

Then:

```powershell
pytest -q
```

Current validation:

```text
Compilation: PASS
Automated tests: PASS
19 tests passed
```

---

# Functional Validation

The system has been validated through direct API and service calls.

## Health

```text
PASS
```

## Patient P001

```text
PASS
```

## Unknown patient P999

```text
404: PASS
```

## Safe patient question

```text
STATUS: success

RESPONSE:
Pending exams: blood pressure monitoring
```

## Unsafe question

```text
STATUS: human_validation_required
```

## Unknown patient

```text
STATUS: patient_not_found
```

---

# Smoke Test

A complete smoke/quality validation can include:

```powershell
python -m py_compile `
    app\main.py `
    app\rag\retriever.py `
    app\rag\rag_qa.py `
    app\llm\assistant_service.py `
    app\llm\assistant_chain.py

pytest -q
```

The current project state has successfully passed this validation.

---

# Known Warning

The test suite currently reports one dependency warning:

```text
DeprecationWarning:
The anyio.abc.BlockingPortal alias is deprecated
```

The warning originates from the installed Starlette/TestClient dependency chain.

It does not currently cause test failures.

Current result:

```text
19 passed, 1 warning
```

---

# Project Status

| Component              | Status   |
| ---------------------- | -------- |
| Backend                | Complete |
| Synthetic Dataset      | Complete |
| Patient Context        | Complete |
| RAG Retrieval          | Complete |
| Entity Detection       | Complete |
| RAG Evaluation         | Complete |
| Safety Layer           | Complete |
| Local Qwen Integration | Complete |
| Assistant Service      | Complete |
| FastAPI Integration    | Complete |
| API Validation         | Complete |
| End-to-End Tests       | Complete |
| Automated Test Suite   | Passing  |

### Current Quality Gate

```text
19 / 19 tests passing
```

---

# QA Engineering Perspective

MedAssist AI was developed not only as an AI application, but also as a practical **AI Quality Engineering** project.

The validation strategy covers multiple layers:

```text
Unit Tests
    |
    v
Integration Tests
    |
    v
RAG Evaluation
    |
    v
End-to-End Tests
    |
    v
API Functional Validation
    |
    v
Quality Gate
```

The project demonstrates testing concerns specific to AI-assisted systems, including:

- Retrieval correctness
- Entity correctness
- Question intent
- Grounded generation
- Safety boundaries
- Patient-context isolation
- Failure handling
- LLM workflow orchestration

---

# Design Principles

## Grounded AI

Generated answers should be based on available evidence instead of unsupported assumptions.

## Explicit Safety Boundaries

Prescription-related requests are intercepted and routed to human validation.

## Entity Awareness

The retrieval layer considers the intended medical entity.

## Testability

Core components are isolated and covered by automated tests.

## Separation of Responsibilities

The application separates:

```text
API
Patient Services
Safety
Clinical Context
Retrieval
LLM
Testing
```

## Fail Safely

When reliable information is unavailable, the system should avoid fabricating an answer.

---

# Future Improvements

Potential next iterations include:

- Docker containerization
- GitHub Actions CI/CD
- Persistent vector database
- Improved embedding models
- Larger instruction-tuned LLM
- Streaming responses
- RAG observability
- Retrieval metrics dashboard
- Hallucination evaluation
- LLM-as-a-judge evaluation
- Prompt versioning
- Model/version tracking
- LangGraph agent workflows
- Structured medical evidence
- Audit logging
- Authentication and authorization
- Role-based access control
- Security testing
- Performance testing
- Load testing
- Production monitoring

---

# Disclaimer

MedAssist AI is an experimental software engineering and AI research project using synthetic patient data.

It is **not a medical device**, does not provide professional medical advice, and must not be used to diagnose, treat, or prescribe for real patients.

All clinical examples are intended for software development, testing, AI engineering and educational purposes.

---

# Author

**Leonardo Mendes de França**

QA Engineer | AI & Automation

Focus areas:

- Quality Engineering
- Test Automation
- Python
- FastAPI
- API Testing
- Playwright
- RAG
- LLM Applications
- AI Quality Engineering
- Automated Testing
- Software Quality

---

# Project Objective

The long-term objective of MedAssist AI is to explore how **AI, RAG, automation and software quality engineering can be combined to build safer and more reliable AI-assisted applications**.

The project follows this principle:

```text
Better AI
    +
Better Retrieval
    +
Better Testing
    +
Better Safety
    =
More Reliable AI Systems
```
