
## RAG System – Hallucination Control & Cost Monitoring Dashboard

This project implements a Retrieval-Augmented Generation (RAG) system focused on reducing hallucinations and controlling LLM cost through re-ranking, self-correction, and detailed token & latency monitoring via a Streamlit dashboard.

---

## Problem This System Solves

Large Language Models can:

Hallucinate answers

Use irrelevant context

Consume excessive tokens (high cost)

## This system addresses those issues by:

Selecting only the most relevant context

Re-ranking retrieved documents

Applying self-correction

Tracking token usage, latency, and ingestion cost

---

## Project Structure

```

rag-system/
├── data/
│   └── loan_faq.csv
├── logs/
├── app.py
├── main.py
├── generator.py
├── retriever.py
├── reranker.py
├── self_corrector.py
├── token_utils.py
├── logger.py
├── metrics.py
├── ingestion_metrics.py
├── requirements.txt

````

---

## Setup

### 1. Clone the repository
```bash
git clone https://github.com/Manuhd/RAG.git

cd RAG/rag_project_rerank
````

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Run the Project

Step 1: 

Run Ingestion (One-Time / On Data Change)

```
py ingest.py
```

Step 2:
```bash
py main.py
```
Step 3:
```
 python -m streamlit run app.py
```
---

##  RAG Pipeline Flow

1. Ingest documents from CSV
2. Generate embeddings
3. Retrieve top-K relevant chunks
4. Re-rank retrieved results
5. Generate final answer
6. Apply self-correction
7. Track metrics and logs

---

## Use Case

* FAQ systems
* Loan / banking Q&A
* Enterprise document search
* Knowledge assistants



---

## Author

**Manu**
GenAI | RAG | Agentic AI
