import streamlit as st
import os
import pandas as pd
import time
from retriever import load_data, build_index, retrieve
from reranker import rerank
from generator import generate_answer
from self_corrector import self_correct
from metrics import hallucination_risk
from logger import save_to_csv
from ingestion_metrics import ingestion_token_stats_from_df
from token_utils import query_token_stats
from sentence_transformers import SentenceTransformer
import numpy as np

embed_model = SentenceTransformer("all-MiniLM-L6-v2")



# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="RAG Hallucination Dashboard",
    layout="wide"
)

st.title(" Gemini RAG – Hallucination Control Dashboard")


# ---------------- LOAD DATA ONCE ----------------
@st.cache_resource
def setup():
    df = load_data()
    index, _ = build_index(df)
    return df, index

df, index = setup()



# ---------------- INGESTION TOKEN METRICS (DYNAMIC) ----------------
st.subheader("Ingestion Token Metrics (One-Time Cost)")

 DEFINE ingestion_stats FIRST
ingestion_stats = ingestion_token_stats_from_df(df)

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Documents(Rows)", ingestion_stats["docs"])

with col2:
    st.metric("Chunks Created", ingestion_stats["chunks"])

with col3:
    st.metric(
        "Total Chunk Tokens",
        f"{ingestion_stats['total_chunk_tokens']:,}"
    )

st.caption("Embedding cost is paid only once during ingestion.")

# ---------------- USER INPUT ----------------
query = st.text_input("Ask a question:")
ask = st.button("Ask")


if ask and query:
    start_time = time.time()   # START latency

    # ---------- Query Embedding ----------
    q_emb = embed_model.encode([query])[0]

    with st.expander("🔎 Query Embedding Info"):
        st.write(f"**Dimension:** {len(q_emb)}")
        st.write(f"**Vector Norm:** {np.linalg.norm(q_emb):.4f}")
        st.write("**Embedding Preview (first 8 values):**")
        st.code(np.round(q_emb[:8,...], 4))

    # ---------- Retrieval ----------
    docs = retrieve(query, df, index)
    reranked_docs = rerank(query, docs)
    MAX_CONTEXT = 5
    SIM_THRESHOLD = 0.5

    num_questions = query.lower().count(" and ") + 1
    num_questions = min(num_questions, MAX_CONTEXT)

    top_context = reranked_docs[
        reranked_docs["similarity"] >= SIM_THRESHOLD
        ].head(num_questions)

    # ---------- Context for LLM ----------

    retrieved_context_text = "\n".join(
        top_context["answer"].tolist()
    )

    # ----------LLM Generation ----------
    answer = generate_answer(retrieved_context_text, query)

    # ---------- Self-correction ----------
    final_answer, corrected, faithfulness = self_correct(
        answer, top_context
    )

    end_time = time.time()     # END latency
    latency = round(end_time - start_time, 2)

    risk = hallucination_risk(faithfulness)

    # ---------- UI ----------
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🤖 Model Answer")
        st.success(final_answer)

        st.subheader("📊 Evaluation Metrics")
        st.metric("Faithfulness Score", round(faithfulness, 2))
        st.metric("Hallucination Risk", risk)
        st.metric("Latency (seconds)", latency)

    with col2:
        st.subheader("📚 Retrieved Context (with Similarity Scores)")

        for _, row in reranked_docs.iterrows():
            c1, c2 = st.columns([4, 1])

            with c1:
                st.write(f"**Q:** {row['question']}")
                st.write(f"**A:** {row['answer']}")

            with c2:
                st.metric("Similarity", row["similarity"])

            st.write("---")

    # ---------- SAVE TO CSV ----------
    save_to_csv(
        question=query,
        answer=final_answer,
        faithfulness=faithfulness,
        hallucination_risk=risk,
        corrected=corrected
    )

    # ---------- TOKEN USAGE (PER QUERY) ----------
    system_prompt = "You are a factual assistant answering only from context."

    token_stats = query_token_stats(
        system_prompt=system_prompt,
        question=query,
        retrieved_context=retrieved_context_text,
        output=final_answer
    )

    st.subheader("🧮 Token Usage (Per Query)")

    t1, t2, t3, t4, t5 = st.columns(5)

    with t1:
        st.metric("System", token_stats["system_tokens"])

    with t2:
        st.metric("Question", token_stats["question_tokens"])

    with t3:
        st.metric("Retrieved", token_stats["retrieved_tokens"])

    with t4:
        st.metric("Output", token_stats["output_tokens"])

    with t5:
        st.metric(
            "Total Tokens",
            token_stats["total_tokens"],
            help="Total tokens consumed for this query"
        )

# ---------------- LOG VIEWER ----------------
st.subheader("📁 Question Answer Logs")

if os.path.exists("logs/qa_logs.csv"):
    logs_df = pd.read_csv("logs/qa_logs.csv")
    st.dataframe(logs_df, width="stretch", hide_index=True)
else:
    st.info("No logs found yet. Ask a question to generate logs.")
