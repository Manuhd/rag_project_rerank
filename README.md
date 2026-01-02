
# RAG System with Re-ranking

This project implements a **Retrieval-Augmented Generation (RAG)** pipeline with
document ingestion, retrieval, re-ranking, self-correction, and evaluation
metrics. The goal is to improve answer quality by selecting the most relevant
context before generation.

---

## Features

- CSV-based document ingestion
- Embedding-based retrieval
- Cross-encoder re-ranking
- Self-correction mechanism
- Token usage tracking
- Ingestion and retrieval metrics
- Structured logging

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
