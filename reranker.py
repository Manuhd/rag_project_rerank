from difflib import SequenceMatcher
"""
def rerank(query, docs):
    docs = docs.copy()  # ✅ FIX: avoid SettingWithCopyWarning

    def score(text):
        return SequenceMatcher(None, query.lower(), text.lower()).ratio()

    docs.loc[:, "rerank_score"] = docs["question"].apply(score)
    return docs.sort_values("rerank_score", ascending=False)
"""

def rerank(query, docs):
    def score(question):
        # dummy reranker logic (replace with cross-encoder later)
        return len(set(query.lower().split()) & set(question.lower().split()))

    docs = docs.copy()
    docs.loc[:, "rerank_score"] = docs["question"].apply(score)

    return docs.sort_values(
        by=["rerank_score", "similarity"],
        ascending=False
    )

