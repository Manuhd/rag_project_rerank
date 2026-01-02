# generator.py

import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

generation_config = {
    "temperature": 0.6,      # creativity control
    "top_p": 0.9,
    "max_output_tokens": 230
}

model = genai.GenerativeModel(
    model_name="gemini-2.5-flash-lite",
    generation_config=generation_config
)
def generate_answer(context_text: str, query: str) -> str:
    """
    context_text: already prepared string (retrieved context)
    query: user question
    """

    prompt = f"""
You are a factual assistant.

Answer ONLY using the context below.
If the answer is not present, say exactly:
"I don't know"

Be clear, slightly conversational, and easy to understand.


Context:
{context_text}

Question:
{query}
"""

    response = model.generate_content(prompt)
    return response.text.strip()
