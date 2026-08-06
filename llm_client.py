'''import os
import requests
import socket


HF_TOKEN = os.getenv("HF_TOKEN")
# Pick a model that is openly available and fast
MODEL = "HuggingFaceH4/zephyr-7b-beta"   # or "google/gemma-2b-it", "microsoft/Phi-3-mini-4k-instruct"

API_URL = f"https://api-inference.huggingface.co/models/{MODEL}"

headers = {
    "Authorization": f"Bearer {HF_TOKEN}",
    "Content-Type": "application/json"
}'''

"""
llm_client.py — Generates answers using a local model (no internet required).
"""

from rag_pipeline import get_llm  # uses the LocalLLM singleton

def generate(query: str, context: str) -> str:
    system = (
        "You are a helpful assistant answering questions about a candidate's CV. "
        "Using only the information in the context below, write a clear, direct answer "
        "in complete sentences. Do not mention 'sources' or use placeholders like [Source 1]. "
        "If the answer isn't in the context, say 'I don't have that information in the CV.'"
    )
    prompt = f"{system}\n\nContext:\n{context}\n\nQuestion: {query}\n\nAnswer:"

    try:
        llm = get_llm()
        return llm.generate(prompt).strip()
    except Exception as e:
        print(f"❌ LLM generation error: {e}")
        return f"Error: {str(e)}"