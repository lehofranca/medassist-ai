# MedAssist AI

AI-assisted clinical information system built with **Python, FastAPI, Retrieval-Augmented Generation (RAG), LangChain, FAISS, Sentence Transformers, PyTorch and a local Qwen language model**.

The project is a **functional proof of concept** focused on combining AI engineering and QA engineering practices in a clinical-information scenario.

> **Important:** MedAssist AI is an educational and technical proof of concept. It is not intended to diagnose patients, prescribe medication, replace healthcare professionals, or make autonomous clinical decisions. All patient data used by the project is synthetic.

---

## 1. Project Overview

MedAssist AI demonstrates an end-to-end architecture in which a user question passes through deterministic application rules, patient-specific context, medical-document retrieval and grounded LLM generation.

The project was designed with a strong focus on:

- Grounded AI responses
- Patient-specific clinical context
- Retrieval quality
- Entity-aware retrieval
- Question-intent awareness
- Medical safety
- Human validation for prescription-related requests
- Automated testing
- RAG-specific evaluation
- API validation
- End-to-end validation
- Explainable retrieval results
- Clear separation between application layers

The central design principle is:

```text
Retrieved evidence > model knowledge
```

The LLM should generate its answer from the context supplied by the application rather than relying on unsupported model knowledge.

---

## 2. Architecture

```text
                         +----------------+
                         |     Client     |
                         +-------+--------+
                                 |
                                 v
                         +----------------+
                         |    FastAPI     |
                         +-------+--------+
                                 |
                                 v
                         +----------------+
                         |  Safety Layer  |
                         +-------+--------+
                                 |
                 +---------------+---------------+
                 |                               |
          Unsafe request                    Safe request
                 |                               |
                 v                               v
     Human Validation Required          Patient Context
                                                 |
                                                 v
                                        +----------------+
                                        |  RAG Retrieval |
                                        +-------+--------+
                                                |
                                                v
                                      Relevant Medical
                                           Evidence
                                                |
                                                v
                                        +----------------+
                                        |   Qwen LLM     |
                                        +-------+--------+
                                                |
                                                v
                                      Grounded Response
```

### Request flow

```text
Question
   |
   v
Safety validation
   |
   +---- Unsafe ----> Human validation required
   |
   v
Patient lookup
   |
   +---- Unknown patient ----> Patient not found
   |
   v
Patient clinical context
   |
   v
RAG retrieval
   |
   v
Relevant medical evidence
   |
   v
Grounded prompt
   |
   v
Qwen model
   |
   v
Response
```

---

## 3. Technology Stack

| Technology            | Purpose                                |
| --------------------- | -------------------------------------- |
| Python 3.11           | Main programming language              |
| FastAPI               | REST API                               |
| Uvicorn               | Application server                     |
| LangChain             | LLM orchestration                      |
| LangChain Core        | Runnable chain implementation          |
| Transformers          | Local model integration                |
| Qwen                  | Local instruction-tuned language model |
| PyTorch               | Model execution                        |
| FAISS                 | Vector similarity search               |
| Sentence Transformers | Embedding generation                   |
| Pytest                | Automated testing                      |
| PowerShell            | Development and functional validation  |

### Current model

```text
Qwen/Qwen2.5-0.5B-Instruct
```

The model is loaded locally through Transformers.

---

## 4. Project Structure

```text
medassist-ai/
│
├── app/
│   ├── __init__.py
│   ├── config.py
│   ├── main.py
│   │
│   ├── llm/
│   │   ├── adapter_inference.py
│   │   ├── assistant_chain.py
│   │   ├── assistant_service.py
│   │   ├── before_after_evaluation.py
│   │   ├── create_smoke_dataset.py
│   │   ├── data_preparation.py
│   │   ├── dataset_validation.py
│   │   ├── evaluate_model.py
│   │   ├── evaluation_pipeline.py
│   │   ├── format_training_dataset.py
│   │   ├── prepare_medquad.py
│   │   ├── split_dataset.py
│   │   ├── training_config.py
│   │   ├── training_formatter.py
│   │   └── training_pipeline.py
│   │
│   ├── rag/
│   │   ├── answer_validator.py
│   │   ├── build_index.py
│   │   ├── clinical_context.py
│   │   ├── context_function.txt
│   │   ├── evaluate_retrieval.py
│   │   ├── patient_service.py
│   │   ├── prompt_function.txt
│   │   ├── rag_qa.py
│   │   ├── retriever.py
│   │   └── specificity_function.txt
│   │
│   └── safety/
│       └── safety_rules.py
│
├── data/
│   ├── evaluation/
│   │   ├── medassist_evaluation_results.json
│   │   ├── medquad_eval.json
│   │   └── rag_retrieval_results.json
│   │
│   ├── processed/
│   │   ├── dataset_split/
│   │   ├── medquad_processed.json
│   │   └── validation_report.json
│   │
│   ├── synthetic/
│   │   └── patients.json
│   │
│   └── training/
│       ├── medquad_training.json
│       ├── train_smoke.json
│       └── validation_formatted.json
│
├── docs/
├── notebooks/
│
├── tests/
│   ├── rag_quality_report.py
│   ├── test_api.py
│   ├── test_assistant_e2e.py
│   ├── test_assistant_service.py
│   ├── test_clinical_context.py
│   ├── test_patient_service.py
│   ├── test_rag_evaluation.py
│   ├── test_rag_retrieval.py
│   ├── test_safety.py
│   └── test_safety_evaluation.py
│
├── .gitignore
├── pytest.ini
├── requirements.txt
└── README.md
```

Generated environments, caches, vector indexes and large model/checkpoint artifacts are intentionally excluded from version control.

---

# 5. Main Components

## 5.1 FastAPI

The API layer exposes the application's main functionality through HTTP endpoints.

Main endpoints:

```text
GET  /
GET  /patients/{patient_id}/context
POST /assistant/ask
```

---

## 5.2 Patient Context

Patient information is maintained separately from the general medical knowledge retrieved by RAG.

Example synthetic patient:

```text
Patient ID: P001
Age: 45
Sex: F
Conditions: hypertension
Medications: losartan
Pending exams: blood pressure monitoring

Recent exams:
- Complete Blood Count
  status=completed
  result=Within reference range
```

The patient dataset is synthetic and exists only for demonstration and testing.

---

## 5.3 Safety Layer

Medical safety is treated as a first-class application concern.

The safety layer evaluates the question before the assistant performs the normal clinical-information flow.

Prescription-related requests are blocked from automatic processing.

Example:

```text
What medication should I take?
```

Expected response:

```json
{
  "status": "human_validation_required",
  "patient_id": "P001",
  "question": "What medication should I take?",
  "message": "Medical prescription-related requests require human validation."
}
```

The unsafe request therefore does not continue through:

```text
Patient Context
      |
      v
RAG
      |
      v
LLM
```

Instead:

```text
Question
   |
   v
Safety Layer
   |
   v
Human Validation Required
```

This establishes a deterministic boundary between informational questions and requests that could result in individualized medical treatment.

---

## 5.4 Patient Validation

Unknown patients are rejected before the assistant produces a clinical response.

Example:

```text
Patient: P999
```

Expected behavior:

```text
404 Not Found
```

The end-to-end tests also verify that the LLM is not unnecessarily invoked for an unknown patient.

---

# 6. RAG Pipeline

The RAG layer retrieves relevant medical information from the indexed knowledge base.

Current indexed corpus:

```text
5,000 medical documents
```

FAISS is used for vector similarity search.

The retrieval process considers multiple signals:

- Semantic similarity
- Entity matching
- Question type
- Specificity
- Final retrieval score

Conceptually:

```text
User Question
     |
     v
Embedding
     |
     v
FAISS Search
     |
     +---- Semantic similarity
     |
     +---- Entity matching
     |
     +---- Question type
     |
     +---- Specificity
     |
     v
Ranked Results
     |
     v
Relevant Evidence
```

---

# 7. Entity-Aware Retrieval

The retrieval layer does not rely exclusively on semantic similarity.

It also evaluates the medical entity referenced by the question.

Examples:

```text
What are the symptoms of hypertension?
        |
        v
hypertension
```

```text
What are the symptoms of high blood pressure?
        |
        v
high blood pressure
```

```text
What are the symptoms of pulmonary hypertension?
        |
        v
pulmonary hypertension
```

This distinction is important because related medical conditions should not automatically be treated as equivalent retrieval targets.

For example:

```text
Hypertension
High Blood Pressure
Pulmonary Hypertension
```

may correspond to different documents and different clinical contexts.

---

# 8. Question Intent

The retrieval system also considers the type of question.

Supported intent examples include:

```text
Symptoms
Diagnosis
Treatment
Causes
Risk factors
Complications
General information
```

This prevents a highly similar document with a different intent from automatically becoming the preferred source.

For example, for:

```text
What are the symptoms of Pulmonary Hypertension?
```

the system should prefer a symptom-specific document over a generic definition or diagnosis document.

---

# 9. Retrieval Quality Example

Example query:

```text
What are the symptoms of pulmonary hypertension?
```

The retrieval system produced:

```text
RESULT 1

Final score:       0.9113
Semantic score:    0.8387
Entity score:      1.0000
Question type:     1.0000
Specificity penalty: 0.0000

Question:
What are the symptoms of Pulmonary Hypertension ?
```

Additional retrieved results included:

```text
What is (are) Pulmonary Hypertension ?

How to diagnose Pulmonary Hypertension ?
```

The context builder selected the direct symptom source as the primary evidence.

The generated answer was grounded in the retrieved medical evidence.

---

# 10. RAG Quality Evaluation

The project includes a dedicated RAG evaluation suite.

Current evaluation cases:

```text
1. Hypertension
2. High Blood Pressure
3. Pulmonary Hypertension
```

Current measured results:

| Metric                   |  Result |
| ------------------------ | ------: |
| Evaluation cases         |       3 |
| Entity accuracy          | 100.00% |
| Top-1 retrieval accuracy | 100.00% |
| Top-3 retrieval accuracy | 100.00% |
| Average retrieval score  |  0.8338 |
| Minimum retrieval score  |  0.7751 |
| Minimum score threshold  |    0.45 |
| Quality Gate             |    PASS |

The dedicated report can be executed with:

```powershell
python tests\rag_quality_report.py
```

The automated RAG evaluation can be executed with:

```powershell
pytest -v tests\test_rag_evaluation.py
```

---

# 11. Grounded Generation

The assistant combines two types of information:

```text
Patient Clinical Context
+
Retrieved Medical Evidence
        |
        v
     Qwen LLM
        |
        v
Grounded Response
```

The model is instructed to use the supplied context as the basis for the response.

This architecture helps reduce unsupported generation and makes the answer more traceable to the application's knowledge sources.

The application also exposes retrieval information during direct RAG execution, including scores and the source question selected as evidence.

---

# 12. Assistant Service

The assistant service orchestrates the main business flow.

Conceptually:

```text
ask_assistant(patient_id, question)
              |
              v
       Validate safety
              |
       +------+------+
       |             |
     Unsafe          Safe
       |             |
       v             v
Human validation   Patient lookup
                         |
                   +-----+-----+
                   |           |
                Unknown       Found
                   |           |
                   v           v
             Not found   Clinical context
                              |
                              v
                         RAG retrieval
                              |
                              v
                         LLM generation
                              |
                              v
                           Response
```

---

# 13. API Examples

## 13.1 Health Check

Request:

```http
GET /
```

Response:

```json
{
  "status": "running",
  "service": "MedAssist AI"
}
```

---

## 13.2 Patient Context

Request:

```http
GET /patients/P001/context
```

Response:

```json
{
  "patient_id": "P001",
  "context": "Patient ID: P001\nAge: 45\nSex: F\nConditions: hypertension\nMedications: losartan\nPending exams: blood pressure monitoring..."
}
```

---

## 13.3 Safe Assistant Question

Request:

```http
POST /assistant/ask
Content-Type: application/json
```

Body:

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

## 13.4 Unsafe Question

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

---

## 13.5 Unknown Patient

Request:

```json
{
  "patient_id": "P999",
  "question": "What are the symptoms of hypertension?"
}
```

Expected result:

```text
404 Not Found
```

The application should not invoke the LLM for a patient that does not exist.

---

# 14. Running the Project

## 14.1 Clone

```powershell
git clone <repository-url>
cd medassist-ai
```

## 14.2 Create Virtual Environment

```powershell
python -m venv .venv
```

## 14.3 Activate

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

## 14.4 Install Dependencies

```powershell
pip install -r requirements.txt
```

---

# 15. Run the API

Start FastAPI with Uvicorn:

```powershell
uvicorn app.main:app --reload
```

API:

```text
http://127.0.0.1:8000
```

Swagger:

```text
http://127.0.0.1:8000/docs
```

---

# 16. Run the RAG + LLM Directly

The complete RAG pipeline can also be executed independently:

```powershell
python app\rag\rag_qa.py
```

Example:

```text
=== MedAssist AI - RAG + LLM ===

Enter your medical question:
What are the symptoms of pulmonary hypertension?
```

The pipeline performs:

```text
Load RAG components
       |
       v
Retrieve relevant documents
       |
       v
Display retrieval scores
       |
       v
Build grounded context
       |
       v
Load tokenizer
       |
       v
Load Qwen model
       |
       v
Generate grounded answer
       |
       v
Display sources
```

---

# 17. Testing Strategy

The project uses a layered testing strategy rather than relying exclusively on end-to-end tests.

```text
                  /\
                 /  \
                / E2E\
               /------\
              /  API   \
             /----------\
            / Integration\
           /--------------\
          /   Unit Tests   \
         /------------------\
```

Current test areas include:

- API
- Patient service
- Clinical context
- Safety rules
- RAG retrieval
- RAG evaluation
- Assistant service
- End-to-end assistant behavior
- Safety evaluation

---

# 18. Automated Test Suite

Run the complete suite:

```powershell
pytest -q
```

Current validated baseline:

```text
....................                                                     [100%]

20 passed, 1 warning
```

The test suite contains:

```text
20 tests
0 failures
```

The warning currently originates from the Starlette/AnyIO dependency stack:

```text
DeprecationWarning:
The anyio.abc.BlockingPortal alias is deprecated
```

It does not currently cause test failures.

---

# 19. Test Files

### API

```text
tests/test_api.py
```

Validates FastAPI behavior and HTTP endpoints.

### Patient Service

```text
tests/test_patient_service.py
```

Validates patient lookup behavior.

### Clinical Context

```text
tests/test_clinical_context.py
```

Validates construction and handling of patient-specific clinical context.

### Safety

```text
tests/test_safety.py
tests/test_safety_evaluation.py
```

Validates deterministic safety behavior and prescription-related escalation.

### Assistant Service

```text
tests/test_assistant_service.py
```

Validates assistant business logic.

### End-to-End

```text
tests/test_assistant_e2e.py
```

Validates the complete assistant flow.

### RAG Retrieval

```text
tests/test_rag_retrieval.py
tests/test_rag_evaluation.py
```

Validates retrieval behavior and evaluation criteria.

### RAG Quality Report

```text
tests/rag_quality_report.py
```

Provides a human-readable retrieval quality report.

---

# 20. Safety Evaluation

The project contains a dedicated safety evaluation suite.

Current scenarios include:

```text
Safe clinical question
        |
        v
success
```

```text
Prescription request
        |
        v
human_validation_required
```

```text
Medication dosage
        |
        v
human_validation_required
```

```text
Treatment request
        |
        v
human_validation_required
```

Run:

```powershell
pytest -v tests\test_safety_evaluation.py
```

The safety suite is designed to verify that requests involving individualized treatment or medication decisions are not automatically processed as ordinary informational questions.

---

# 21. Functional Validation

The project also includes a functional validation gate covering the major application behaviors.

A representative validation sequence is:

```text
========== MEDASSIST AI FUNCTIONAL GATE ==========

[1/5] HEALTH CHECK
PASS

[2/5] PATIENT CONTEXT - P001
PASS

[3/5] UNKNOWN PATIENT - P999
Expected 404: PASS

[4/5] AUTOMATED TEST SUITE
20 passed

[5/5] ASSISTANT SERVICE

SAFE PATIENT QUESTION
STATUS: success

UNSAFE QUESTION
STATUS: human_validation_required

UNKNOWN PATIENT
STATUS: patient_not_found

========== FUNCTIONAL GATE COMPLETE ==========
```

This validates not only isolated functions but the expected application behavior.

---

# 22. Recommended Development Workflow

The project follows a quality-gate workflow:

```text
Implement
   |
   v
Compile
   |
   v
Unit Tests
   |
   v
RAG Evaluation
   |
   v
Safety Evaluation
   |
   v
Integration Tests
   |
   v
E2E Tests
   |
   v
API Validation
   |
   v
Functional Gate
```

Example compilation check:

```powershell
python -m py_compile app\main.py
```

Example test execution:

```powershell
pytest -q
```

---

# 23. Quality Gates

A change should be considered ready for integration when the relevant validation gates pass:

```text
[PASS] Python environment
[PASS] Compilation
[PASS] Unit tests
[PASS] Integration tests
[PASS] RAG evaluation
[PASS] Safety evaluation
[PASS] E2E tests
[PASS] API validation
[PASS] Functional validation
```

This approach applies traditional QA principles to an AI-powered application.

---

# 24. QA Engineering Perspective

MedAssist AI was intentionally designed as both an **AI engineering** and **QA engineering** project.

Traditional QA:

```text
Traditional QA
    |
    +-- API Testing
    +-- Integration Testing
    +-- Unit Testing
    +-- E2E Testing
    +-- Negative Testing
    +-- Regression Testing
```

AI Quality:

```text
AI Quality
    |
    +-- Retrieval Evaluation
    +-- Entity Matching
    +-- Semantic Relevance
    +-- Question Intent
    +-- Specificity
    +-- Grounding
    +-- Safety Validation
    +-- Hallucination Risk Reduction
```

The project demonstrates that testing AI systems requires more than verifying HTTP status codes or deterministic outputs.

The quality strategy must also evaluate:

```text
Was the right evidence retrieved?
        |
        v
Was the correct medical entity identified?
        |
        v
Was the question intent respected?
        |
        v
Was the response grounded?
        |
        v
Was an unsafe request blocked?
```

---

# 25. Data Strategy

The project uses multiple data stages:

```text
Raw Data
   |
   v
Processing
   |
   v
Validation
   |
   v
Dataset Split
   |
   +---- Train
   +---- Validation
   +---- Test
   |
   v
Evaluation / Training
```

The repository contains selected processed, evaluation, synthetic and training artifacts that support project inspection.

Large generated datasets, raw data and local model artifacts are excluded through `.gitignore`.

This keeps the Git repository suitable for academic evaluation without committing unnecessary binary or generated artifacts.

---

# 26. Model and Training Pipeline

The repository also contains scripts supporting dataset preparation and model evaluation/training workflows.

Relevant modules include:

```text
prepare_medquad.py
data_preparation.py
dataset_validation.py
format_training_dataset.py
split_dataset.py
training_formatter.py
training_config.py
training_pipeline.py
evaluate_model.py
evaluation_pipeline.py
before_after_evaluation.py
create_smoke_dataset.py
adapter_inference.py
```

The current functional assistant uses the local Qwen instruction model.

The training/evaluation utilities are maintained separately from the runtime assistant flow so that experimentation does not unnecessarily complicate the production-like application path.

---

# 27. Repository Hygiene

The repository intentionally excludes generated or environment-specific artifacts.

Examples:

```text
.venv/
.pytest_cache/
__pycache__/
data/raw/
data/processed/faiss_index/
data/processed/medassist-qwen-lora/
data/processed/medassist-qwen-smoke/
data/processed/medquad_full.json
data/processed/dataset_split/train.json
data/training/train_formatted.json
```

This makes the repository easier to clone, inspect and evaluate.

---

# 28. Current Validation Status

Current validated baseline:

| Area                     | Status |
| ------------------------ | ------ |
| Python 3.11 environment  | PASS   |
| FastAPI                  | PASS   |
| LangChain Core           | PASS   |
| FAISS                    | PASS   |
| Compilation              | PASS   |
| Automated tests          | PASS   |
| Safety validation        | PASS   |
| Patient context          | PASS   |
| Unknown patient handling | PASS   |
| Assistant service        | PASS   |
| End-to-end flow          | PASS   |
| RAG retrieval            | PASS   |
| Entity detection         | PASS   |
| RAG quality gate         | PASS   |
| API health check         | PASS   |
| Local Qwen inference     | PASS   |

Current automated baseline:

```text
20 passed
0 failed
1 dependency warning
```

RAG baseline:

```text
5,000 indexed medical documents

Entity accuracy:          100.00%
Top-1 retrieval accuracy: 100.00%
Top-3 retrieval accuracy: 100.00%
Average retrieval score:  0.8338
Minimum retrieval score:  0.7751
Quality Gate:             PASS
```

---

# 29. Known Warning

The test suite currently reports one dependency warning:

```text
DeprecationWarning:
The anyio.abc.BlockingPortal alias is deprecated
```

The warning originates from the installed Starlette/AnyIO dependency stack.

Current impact:

```text
Test failures: 0
```

Therefore it does not currently block the project's quality gate.

---

# 30. Limitations

MedAssist AI is currently a proof of concept.

Current limitations include:

- Synthetic patient data only
- Local language model
- Limited model size
- Limited medical knowledge base
- No authentication layer
- No production database
- No production observability stack
- No formal clinical validation
- No regulatory certification
- No real-world patient deployment
- Limited safety-rule coverage
- No formal clinical decision-support validation

These limitations are intentional and should be considered before any discussion of production deployment.

---

# 31. Future Improvements

## AI / RAG

- Improve retrieval evaluation metrics
- Add MRR
- Add Recall@K
- Add Precision@K
- Introduce reranking
- Improve chunking strategy
- Add metadata filtering
- Add source confidence
- Improve hallucination detection
- Add answer faithfulness evaluation

## LLM

- Experiment with larger local models
- Compare different instruction-tuned models
- Add structured output
- Add response validation
- Expand model evaluation datasets
- Evaluate latency and resource consumption

## Safety

- Expand safety rules
- Add medical-risk classification
- Add escalation workflows
- Add audit logging
- Add explicit human-in-the-loop approval
- Add broader adversarial safety tests

## QA / Automation

- Expand API integration tests
- Add contract testing
- Add performance testing
- Add negative testing
- Add security testing
- Add CI/CD pipeline
- Add automated quality gates
- Add regression baselines for retrieval quality

## Infrastructure

- Docker
- PostgreSQL
- Production vector database
- Redis
- Observability
- CI/CD
- Cloud deployment

---

# 32. Academic Relevance

The project demonstrates the integration of several areas relevant to modern software and AI engineering:

```text
Software Engineering
        +
Artificial Intelligence
        +
Natural Language Processing
        +
Retrieval-Augmented Generation
        +
Large Language Models
        +
Quality Assurance
        +
Automated Testing
        +
AI Safety
```

From a QA perspective, the project extends conventional testing concepts into AI-specific validation.

Instead of validating only:

```text
Input -> Expected Output
```

the project also evaluates:

```text
Input
  |
  +--> Safety decision
  |
  +--> Patient context
  |
  +--> Entity extraction
  |
  +--> Retrieval relevance
  |
  +--> Question intent
  |
  +--> Evidence grounding
  |
  +--> Generated response
```

This is the main engineering objective of the project.

---

# 33. Project Status

```text
Status: FUNCTIONAL PROOF OF CONCEPT
```

Current baseline:

```text
FastAPI                         Operational
Patient context                Operational
Safety layer                   Operational
RAG retrieval                  Operational
5,000 indexed documents        Validated
Entity-aware retrieval         Validated
Question intent                Validated
Local Qwen model               Operational
Grounded generation            Operational
Automated tests                20 passed
RAG quality gate               PASS
Safety evaluation              PASS
E2E assistant flow             PASS
API validation                 PASS
```

---

# 34. Disclaimer

**MedAssist AI is an educational and technical proof of concept.**

It is not intended to:

- Diagnose medical conditions
- Prescribe medication
- Replace healthcare professionals
- Make autonomous clinical decisions
- Serve as a production clinical decision-support system

All patient information included in the project is synthetic.

Any future production implementation would require, at minimum:

- Formal clinical validation
- Security controls
- Privacy protections
- Appropriate data governance
- Regulatory compliance
- Monitoring
- Human oversight
- Comprehensive safety validation
- Production-grade infrastructure

---

## Author

**Leonardo França**

MedAssist AI — AI / RAG / QA Engineering Academic Project
