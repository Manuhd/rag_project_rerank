import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")

def load_data():
    df = pd.read_csv("data/loan_faq.csv")
    return df

def build_index(df):
    embeddings = model.encode(df["question"].tolist())
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))
    return index, embeddings
"""
def retrieve(query, df, index, top_k=3):
    q_emb = model.encode([query])
    _, indices = index.search(np.array(q_emb), top_k)
    return df.iloc[indices[0]]
"""
def retrieve(query, df, index, top_k=3):
    q_emb = model.encode([query])
    distances, indices = index.search(np.array(q_emb), top_k)

    rows = []
    for idx, dist in zip(indices[0], distances[0]):
        rows.append({
            "question": df.iloc[idx]["question"],
            "answer": df.iloc[idx]["answer"],
            "similarity": round(1 / (1 + dist), 3)
        })

    return pd.DataFrame(rows)
