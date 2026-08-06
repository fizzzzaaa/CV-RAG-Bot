"""
rag_pipeline.py — The core RAG engine.
Handles chunking, embedding, FAISS indexing, and retrieval.
"""

import numpy as np
import faiss
import re
from sentence_transformers import SentenceTransformer
from typing import List, Tuple
from dataclasses import dataclass
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch


class LocalLLM:
    def __init__(self, model_name="HuggingFaceTB/SmolLM-135M-Instruct"):   # ← smaller default
        print(f"Loading local LLM: {model_name}...")
        
        # Use dtype based on hardware
        self.dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        
        # Load model (much smaller, so no special hacks needed)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            dtype=self.dtype,
            low_cpu_mem_usage=True,
        )
        
        # Move to GPU if available
        if torch.cuda.is_available():
            self.model = self.model.to('cuda')
        else:
            self.model = self.model.to('cpu')
        
        # Create pipeline – device=0 for GPU, -1 for CPU
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if torch.cuda.is_available() else -1,
        )
    
    def generate(self, prompt, max_new_tokens=512):    # ← must be indented to match __init__ above
        messages = [{"role": "user", "content": prompt}]
        outputs = self.pipe(
            messages,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            return_full_text=False,
        )

        print("DEBUG outputs type:", type(outputs))
        print("DEBUG outputs value:", outputs)

        if isinstance(outputs, list) and len(outputs) > 0 and isinstance(outputs[0], dict) and 'generated_text' in outputs[0]:
            result = outputs[0]['generated_text']
            if isinstance(result, list):
                return result[-1]['content']
            return result

        if isinstance(outputs, str):
            return outputs

        if isinstance(outputs, list) and len(outputs) > 0 and isinstance(outputs[0], str):
            return outputs[0]

        return str(outputs)


# ── Constants ──────────────────────────────────────────────────────────────────
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # Fast, 384-dim, free
CHUNK_SIZE = 300        # characters per chunk
CHUNK_OVERLAP = 60      # overlap between chunks to preserve context
TOP_K = 4              # how many chunks to retrieve


@dataclass
class Chunk:
    """A single text chunk with metadata."""
    text: str
    source: str          # e.g. "CV", "Notes"
    chunk_id: int
    start_char: int


class RAGPipeline:
    """
    Production-ready RAG pipeline.

    Flow:
      1. Ingest documents → split into overlapping chunks
      2. Embed all chunks with sentence-transformers
      3. Build a FAISS flat-L2 index (cosine via normalized vectors)
      4. At query time: embed query → FAISS ANN search → return top-k chunks
    """

    def __init__(self):
        print("Loading embedding model…")
        self.encoder = SentenceTransformer(EMBEDDING_MODEL)
        self.embedding_dim = self.encoder.get_sentence_embedding_dimension()

        self.index: faiss.IndexFlatIP | None = None  # inner-product == cosine on normed vecs
        self.chunks: List[Chunk] = []

    # ── Ingestion ──────────────────────────────────────────────────────────────

    def ingest(self, documents: dict[str, str]) -> None:
        """
        Ingest a dict of {source_name: raw_text} and build the FAISS index.
        Call once at startup.
        """
        self.chunks = []
        for source, text in documents.items():
            source_chunks = self._split_text(text, source)
            self.chunks.extend(source_chunks)

        print(f"Ingested {len(self.chunks)} chunks from {list(documents.keys())}")
        self._build_index()

    def _split_text(self, text: str, source: str) -> List[Chunk]:
        """
        Sentence-aware chunking with overlap.
        Splits on sentence boundaries first, then enforces CHUNK_SIZE.
        """
        # Normalize whitespace
        text = re.sub(r"\n{3,}", "\n\n", text.strip())

        # Split into sentences (simple but effective)
        sentences = re.split(r"(?<=[.!?])\s+", text)

        chunks: List[Chunk] = []
        current = ""
        current_start = 0
        char_pos = 0

        for sentence in sentences:
            if len(current) + len(sentence) > CHUNK_SIZE and current:
                chunks.append(Chunk(
                    text=current.strip(),
                    source=source,
                    chunk_id=len(chunks),
                    start_char=current_start,
                ))
                # Overlap: keep last CHUNK_OVERLAP chars
                overlap_text = current[-CHUNK_OVERLAP:] if len(current) > CHUNK_OVERLAP else current
                current = overlap_text + " " + sentence
                current_start = char_pos - len(overlap_text)
            else:
                if not current:
                    current_start = char_pos
                current += (" " if current else "") + sentence

            char_pos += len(sentence) + 1  # +1 for the space we split on

        if current.strip():
            chunks.append(Chunk(
                text=current.strip(),
                source=source,
                chunk_id=len(chunks),
                start_char=current_start,
            ))

        return chunks

    def _build_index(self) -> None:
        """Embed all chunks and build FAISS IndexFlatIP (cosine similarity)."""
        texts = [c.text for c in self.chunks]

        print(f"Embedding {len(texts)} chunks…")
        embeddings = self.encoder.encode(
            texts,
            batch_size=64,
            show_progress_bar=True,
            normalize_embeddings=True,   # ← L2 norm → inner product == cosine
            convert_to_numpy=True,
        ).astype(np.float32)

        self.index = faiss.IndexFlatIP(self.embedding_dim)
        self.index.add(embeddings)
        print(f"FAISS index built: {self.index.ntotal} vectors @ dim={self.embedding_dim}")

    def format_context_for_llm(self, retrieved: List[Tuple[Chunk, float]]) -> str:
        """Plain context for the LLM prompt — no source tags, just content."""
        if not retrieved:
            return "No relevant context found."
        return "\n\n".join(chunk.text for chunk, _ in retrieved)

    # ── Retrieval ──────────────────────────────────────────────────────────────

    def retrieve(self, query: str, top_k: int = TOP_K) -> List[Tuple[Chunk, float]]:
        """
        Retrieve the top-k most relevant chunks for a query.

        Returns: list of (Chunk, similarity_score) tuples, sorted desc by score.
        """
        if self.index is None or self.index.ntotal == 0:
            return []

        query_vec = self.encoder.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype(np.float32)

        scores, indices = self.index.search(query_vec, min(top_k, len(self.chunks)))

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0:  # FAISS returns -1 for empty slots
                results.append((self.chunks[idx], float(score)))

        return results

    def format_context(self, retrieved: List[Tuple[Chunk, float]]) -> str:
        """Format retrieved chunks into a context string for the LLM prompt."""
        if not retrieved:
            return "No relevant context found."

        parts = []
        for i, (chunk, score) in enumerate(retrieved, 1):
            parts.append(
                f"[Source {i} — {chunk.source} | relevance: {score:.2f}]\n{chunk.text}"
            )

        return "\n\n---\n\n".join(parts)


# ── Singleton (loaded once at import) ─────────────────────────────────────────
_pipeline: RAGPipeline | None = None

def get_pipeline() -> RAGPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = RAGPipeline()
    return _pipeline

_llm: LocalLLM | None = None

def get_llm() -> LocalLLM:
    global _llm
    if _llm is None:
        _llm = LocalLLM()
    return _llm


