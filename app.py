"""
app.py — Gradio interface for the CV RAG Bot.
Deploy this on Hugging Face Spaces (Gradio SDK).

Space configuration:
  - SDK: Gradio
  - Python: 3.10+
  - Secrets: HF_TOKEN (your Hugging Face API token)
"""

import gradio as gr
import PyPDF2
from llm_client import generate
from cv_data import CV_TEXT, ADDITIONAL_NOTES
from rag_pipeline import get_pipeline, get_llm


# ── Startup: load and index the CV ────────────────────────────────────────────
print("Initializing RAG pipeline…")
default_pipeline = get_pipeline()
default_pipeline.ingest({
    "CV / Resume": CV_TEXT,
    "Personal Notes": ADDITIONAL_NOTES,
})
print("Ready!")

def extract_text_from_pdf(file_path):
    text = ""
    try:
        with open(file_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                text += page.extract_text() or ""
        return text
    except Exception as e:
        return f"ERROR: {str(e)}"
    
# ── State ──────────────────────────────────────────────────────────────────────
# We keep last retrieved sources in a mutable container so the Sources tab
# can display them without threading issues.
_last_sources: list[dict] = []

# ── Suggested questions ────────────────────────────────────────────────────────
EXAMPLE_QUESTIONS = [
    "What is your experience with RAG pipelines?",
    "What ML infrastructure have you worked with?",
    "Tell me about your biggest technical challenge.",
    "What are your Python and cloud skills?",
    "What is your educational background?",
    "When can you start and what are your salary expectations?",
    "What open-source projects have you built?",
    "What conferences or events have you spoken at?",
]

# ── Core chat function ─────────────────────────────────────────────────────────
_last_sources = []

def chat(message: int, history: list, pipeline):
    """RAG pipeline using the given pipeline instance."""
    if pipeline is None:
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": "⚠️ Please upload a CV PDF first."})
        return "", history, gr.update()

    if not message.strip():
        return "", history, gr.update()

    retrieved = pipeline.retrieve(message, top_k=4)
    context = pipeline.format_context(retrieved)
    answer = generate(message, context)

    global _last_sources
    _last_sources = [
        {
            "rank": i + 1,
            "source": chunk.source,
            "score": f"{score:.3f}",
            "text": chunk.text,
        }
        for i, (chunk, score) in enumerate(retrieved)
    ]

    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": answer})
    return "", history, gr.update(value=get_sources_html())

pipeline = get_pipeline()
llm = get_llm()


def get_sources_html() -> int:
    """Render retrieved sources as styled HTML for the Sources tab."""
    if not _last_sources:
        return """
        <div style="padding: 24px; color: #6b7280; text-align: center; font-family: sans-serif;">
            <p style="font-size: 1.1em;">💬 Ask a question to see which CV sections were retrieved.</p>
        </div>
        """

    cards = ""
    for src in _last_sources:
        score_pct = int(float(src["score"]) * 100)
        bar_color = "#10b981" if score_pct > 70 else "#f59e0b" if score_pct > 50 else "#ef4444"

        cards += f"""
        <div style="
            background: #1e1e2e;
            border: 1px solid #313244;
            border-radius: 10px;
            padding: 16px 20px;
            margin-bottom: 14px;
            font-family: 'JetBrains Mono', 'Fira Code', monospace;
        ">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                <span style="
                    background: #313244;
                    color: #cdd6f4;
                    padding: 2px 10px;
                    border-radius: 20px;
                    font-size: 0.75em;
                    font-weight: 700;
                    letter-spacing: 0.05em;
                ">CHUNK #{src['rank']} — {src['source'].upper()}</span>
                <div style="display: flex; align-items: center; gap: 8px;">
                    <div style="
                        width: 80px; height: 6px;
                        background: #313244;
                        border-radius: 3px;
                        overflow: hidden;
                    ">
                        <div style="
                            width: {score_pct}%;
                            height: 100%;
                            background: {bar_color};
                            border-radius: 3px;
                        "></div>
                    </div>
                    <span style="color: {bar_color}; font-size: 0.8em; font-weight: 700;">{src['score']}</span>
                </div>
            </div>
            <p style="
                color: #cdd6f4;
                font-size: 0.82em;
                line-height: 1.7;
                margin: 0;
                white-space: pre-wrap;
                background: #181825;
                padding: 10px 14px;
                border-radius: 6px;
                border-left: 3px solid {bar_color};
            ">{src['text']}</p>
        </div>
        """

    return f"""
    <div style="padding: 4px 0;">
        <p style="
            font-family: sans-serif;
            color: #6b7280;
            font-size: 0.8em;
            margin-bottom: 16px;
        ">
            {len(_last_sources)} chunks retrieved via FAISS cosine similarity search
            (model: <code>sentence-transformers/all-MiniLM-L6-v2</code>)
        </p>
        {cards}
    </div>
    """
    pass


# ── Gradio UI ──────────────────────────────────────────────────────────────────
THEME = gr.themes.Base(
    primary_hue="violet",
    secondary_hue="slate",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
    font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "monospace"],
).set(
    body_background_fill="#0f0f17",
    body_background_fill_dark="#0f0f17",
    block_background_fill="#1e1e2e",
    block_border_color="#313244",
    button_primary_background_fill="#7c3aed",
    button_primary_background_fill_hover="#6d28d9",
    button_primary_text_color="white",
    input_background_fill="#181825",
    input_border_color="#313244",
    checkbox_background_color="#181825",
)

CSS = """
#chat-title {
    text-align: center;
    padding: 20px 0 8px;
}
#chat-title h1 {
    font-size: 1.8em;
    font-weight: 800;
    background: linear-gradient(135deg, #a78bfa, #60a5fa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 4px;
}
#chat-title p {
    color: #6b7280;
    font-size: 0.9em;
}
.example-btn {
    font-size: 0.78em !important;
}
#chatbot { min-height: 300px; }
#sources-html { min-height: 300px; }
.tab-nav button { font-weight: 600; }
footer { display: none !important; }
"""

def on_file_upload(file_path, current_pipeline):
    if file_path is None:
        return current_pipeline, "📂 *No CV loaded. Please upload a PDF to start chatting.*"
    
    text = extract_text_from_pdf(file_path)
    if text.startswith("ERROR"):
        return current_pipeline, f"❌ {text}"
    
    new_pipeline = get_pipeline()
    new_pipeline.ingest({"Uploaded CV": text})
    return new_pipeline, "✅ CV loaded successfully! You can now ask questions."

def respond(message, history, pipeline):
    new_msg, new_history, sources_update = chat(message, history, pipeline)
    return new_msg, new_history, sources_update

with gr.Blocks(theme=THEME, css=CSS, title="CV RAG Bot") as demo:

    # Header
    with gr.Column(elem_id="chat-title"):
        gr.HTML("<h1>🤖 Chat with Alex's CV</h1><p>Powered by RAG · FAISS · sentence-transformers · Llama 3 via Hugging Face</p>")

    with gr.Tabs():
        with gr.Tab("💬 Chat"):
            # Upload row
            with gr.Row():
                file_upload = gr.File(label="📄 Upload your CV (PDF)", file_types=[".pdf"], type="filepath")
                upload_status = gr.Markdown("📂 *No CV loaded. Please upload a PDF to start chatting.*")

            # Pipeline state (shared across chat)
            pipeline_state = gr.State(value=default_pipeline)   # start with default

            # Chatbot – now with larger height
            chatbot = gr.Chatbot(
    height=300, elem_id="chatbot", label="",
       # ← add this
    avatar_images=(None, "https://huggingface.co/datasets/huggingface/brand-assets/resolve/main/hf-logo.svg"),
)

            # Input row
            with gr.Row():
                msg = gr.Textbox(placeholder="Ask anything about Alex's background…", scale=5, container=False, autofocus=True)
                send_btn = gr.Button("Send →", variant="primary", scale=1, min_width=90)

            with gr.Row():
                clear_btn = gr.Button("🗑 Clear chat", variant="secondary", size="sm")

            # Suggested questions (simplified)
            example_buttons = []
            for q in ["What is your experience with RAG?", "Tell me about your biggest challenge.", "What are your Python skills?", "When can you start?"]:
                btn = gr.Button(q, size="sm")
                btn.click(fn=lambda _, q=q: q, inputs=[msg], outputs=[msg])
                example_buttons.append(btn)


        # Tab 2: Retrieved Sources
        with gr.Tab("🔍 Sources"):
            refresh_btn = gr.Button("↻ Refresh sources", variant="secondary", size="sm")
            sources_html = gr.HTML(
                value=get_sources_html(),
                elem_id="sources-html",
            )

        # Tab 3: About
        with gr.Tab("ℹ️ Architecture"):
            gr.Markdown("""
            ## How this RAG Bot works

            ```
            User query
                │
                ▼
            ┌─────────────────────────────────────────────┐
            │   sentence-transformers/all-MiniLM-L6-v2    │  ← Embedding model (384-dim)
            │   Encodes query into a dense vector          │
            └───────────────────┬─────────────────────────┘
                                │  query vector
                                ▼
            ┌─────────────────────────────────────────────┐
            │           FAISS IndexFlatIP                  │  ← Vector DB (in-memory)
            │   ANN search: cosine similarity over         │    ~50 chunks from CV + notes
            │   L2-normalized embeddings                   │
            └───────────────────┬─────────────────────────┘
                                │  top-4 chunks + scores
                                ▼
            ┌─────────────────────────────────────────────┐
            │       Prompt Builder                         │
            │   System prompt + context + query            │
            │   → Llama 3 / Mistral chat format            │
            └───────────────────┬─────────────────────────┘
                                │  formatted prompt
                                ▼
            ┌─────────────────────────────────────────────┐
            │    HF Inference API (serverless)             │  ← No local GPU needed
            │    meta-llama/Meta-Llama-3-8B-Instruct       │
            │    (fallback: Mistral-7B-Instruct-v0.3)      │
            └───────────────────┬─────────────────────────┘
                                │  grounded answer
                                ▼
                         User sees response
                    + Sources tab shows chunks
            ```

            ### Key design decisions

            | Choice | Why |
            |--------|-----|
            | `all-MiniLM-L6-v2` | 384-dim, fast, free — perfect for small corpora |
            | `FAISS IndexFlatIP` | Exact search on normalized vectors = cosine similarity |
            | Sentence-aware chunking | Avoids splitting mid-sentence; preserves semantic units |
            | `CHUNK_OVERLAP = 60` | Prevents losing context at chunk boundaries |
            | `temperature = 0.3` | Low randomness → more factual, less hallucination |
            | HF Inference API | Zero infrastructure cost; no GPU needed |
            | Fallback model | Ensures availability even without Llama access |

            ### Why RAG beats fine-tuning here
            Fine-tuning would require updating weights every time the CV changes.
            RAG lets you **update the knowledge base** (just `pipeline.ingest()` again)
            without touching the model. It also gives you **source attribution** — you
            can see *exactly* which text the model used.
            """)

    # ── Event handlers ────────────────────────────────────────────────────────
    def submit(message, history):
        return chat(message, history)
    
    def respond(message, history, pipeline):
        new_msg, new_history, sources_update = chat(message, history, pipeline)
        return new_msg, new_history, sources_update
    
    msg.submit(
    fn=respond,
    inputs=[msg, chatbot, pipeline_state],
    outputs=[msg, chatbot, sources_html]
    )

    send_btn.click(
    fn=respond,
    inputs=[msg, chatbot, pipeline_state],
    outputs=[msg, chatbot, sources_html]
    )
    
    clear_btn.click(lambda: ([], "", gr.update()), None, [chatbot, msg, sources_html])

# 4. Refresh sources
    refresh_btn.click(
    fn=get_sources_html,
    inputs=None,
    outputs=sources_html
    )
    chatbot.change(get_sources_html, None, sources_html)
    file_upload.change(
        fn=on_file_upload,
        inputs=[file_upload, pipeline_state],
        outputs=[pipeline_state, upload_status]
    )

""""msg.submit(respond, [msg, chatbot, pipeline_state], [msg, chatbot, sources_html])
    send_btn.click(respond, [msg, chatbot, pipeline_state], [msg, chatbot, sources_html])
    clear_btn.click(lambda: ([], "", gr.update()), None, [chatbot, msg, sources_html])
    refresh_btn.click(get_sources_html, None, sources_html) """   

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,          # ← add this
        show_error=True,
    )