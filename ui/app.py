# ui/app.py — Streamlit Frontend
#
# Run with: streamlit run ui/app.py


import sys
import os

# Suppress noisy TensorFlow and tokenizer warnings before any imports
os.environ["TF_CPP_MIN_LOG_LEVEL"]   = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"]  = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

import warnings
warnings.filterwarnings("ignore")

# Add project root to Python path so "from pipeline import ..." works
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import tempfile
import time
from pathlib import Path

# Page Config 
st.set_page_config(
    page_title="AI Document Intelligence",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CSS
st.markdown("""
<style>
    .main { background-color: #0F1117; }
    [data-testid="stSidebar"] {
        background-color: #1A1A2E;
        border-right: 1px solid #2D3748;
    }
    .answer-box {
        background: #0F2940;
        border-left: 4px solid #3B82F6;
        border-radius: 8px;
        padding: 16px 20px;
        margin: 10px 0;
        color: #E2E8F0;
        font-size: 15px;
        line-height: 1.7;
    }
    .source-tag {
        background: #1E3A5F;
        border: 1px solid #3B82F6;
        border-radius: 20px;
        padding: 3px 12px;
        font-size: 12px;
        color: #93C5FD;
        display: inline-block;
        margin: 3px;
    }
    .info-card {
        background: #1E293B;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 16px;
        margin: 8px 0;
    }
    .chat-user {
        background: #1E3A5F;
        border-radius: 12px 12px 4px 12px;
        padding: 12px 16px;
        margin: 8px 0;
        max-width: 85%;
        margin-left: auto;
        color: #E2E8F0;
    }
    .chat-assistant {
        background: #1A2744;
        border-radius: 4px 12px 12px 12px;
        padding: 12px 16px;
        margin: 8px 0;
        max-width: 95%;
        color: #E2E8F0;
        border-left: 3px solid #3B82F6;
    }
    #MainMenu { visibility: hidden; }
    footer    { visibility: hidden; }
    header    { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# Session State

def init_session():
    """
    Initialize all session state variables.
    Streamlit reruns the script on every user action, but values stored
    in st.session_state survive those reruns.
    """
    if "pipeline" not in st.session_state:
        st.session_state.pipeline = None

    if "pipeline_initialized" not in st.session_state:
        st.session_state.pipeline_initialized = False

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if "active_doc" not in st.session_state:
        st.session_state.active_doc = None

    if "indexed_files" not in st.session_state:
        st.session_state.indexed_files = []


def get_pipeline():
    """
    Get or create the DocumentPipeline instance.
    Stored in session_state so it survives Streamlit reruns.
    The pipeline holds all_chunks in memory, which BM25 needs.
    """
    if not st.session_state.pipeline_initialized:
        with st.spinner("Initializing AI pipeline..."):
            try:
                from pipeline import DocumentPipeline
                st.session_state.pipeline = DocumentPipeline()
                st.session_state.pipeline_initialized = True
            except Exception as e:
                st.error(f"Failed to initialize pipeline: {e}")
                import traceback
                st.code(traceback.format_exc())
                return None
    return st.session_state.pipeline



# Sidebar

def render_sidebar():
    with st.sidebar:
        st.markdown("""
        <div style='text-align:center; padding:20px 0;'>
            <div style='font-size:18px; font-weight:bold; color:#E2E8F0;'>
                Doc Intelligence
            </div>
            <div style='font-size:12px; color:#64748B; margin-top:4px;'>
                AI-Powered Document Q&A
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.divider()

        # File Upload
        st.markdown("#### Upload Document")

        uploaded_file = st.file_uploader(
            "Choose a file",
            type=["pdf", "docx", "doc", "txt", "xlsx", "xls", "csv",
                  "png", "jpg", "jpeg"],
            help="Supported: PDF, Word, Excel, CSV, TXT, Images",
        )

        if uploaded_file:
            size_mb = uploaded_file.size / (1024 * 1024)
            st.caption(f"{uploaded_file.name} ({size_mb:.1f} MB)")

            if st.button("Index Document", type="primary",
                         use_container_width=True):
                _index_document(uploaded_file)

        st.divider()

        #  Indexed Documents 
        st.markdown("#### Indexed Documents")

        pipeline = get_pipeline()
        if pipeline:
            docs = pipeline.get_document_list()
            if not docs:
                st.caption("No documents indexed yet.")
            else:
                for doc_name in docs:
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        is_active = st.session_state.active_doc == doc_name
                        label = f"{doc_name} (Current)" if is_active else doc_name
                        if st.button(label, key=f"select_{doc_name}",
                                     use_container_width=True):
                            st.session_state.active_doc = doc_name
                            st.session_state.chat_history = []
                            pipeline.reset_conversation()
                            st.rerun()
                    with col2:
                        if st.button("Remove", key=f"del_{doc_name}",
                                     help="Remove document"):
                            pipeline.remove_document(doc_name)
                            if st.session_state.active_doc == doc_name:
                                st.session_state.active_doc = None
                                st.session_state.chat_history = []
                            st.rerun()

        st.divider()

        # Pipeline Status 
        st.markdown("#### Pipeline Status")
        _show_pipeline_status()

        st.divider()

        # Config 
        with st.expander("Configuration"):
            from config import Config
            st.caption(f"**Backend:** `{Config.get_backend_name()}`")
            st.caption(f"**LLM:** `{Config.get_llm_model()}`")
            st.caption(f"**VLM:** `{Config.get_vlm_model()}`")
            st.caption(f"**Embedder:** `{Config.EMBEDDING_MODEL}`")
            st.caption(f"**Chunk Size:** {Config.CHUNK_SIZE} tokens")
            st.caption(f"**Active Doc:** `{st.session_state.active_doc or 'None'}`")


def _index_document(uploaded_file):
    """
    Save the uploaded file to a temp path and run the indexing pipeline.

    IMPORTANT: We pass original_filename=uploaded_file.name to pipeline.index()
    so the ChromaDB collection is stored under the real filename (e.g. "report.pdf"),
    NOT the temp path name (e.g. "tmpXXXX.pdf").
    This ensures query() can find the collection later using active_doc.
    """
    pipeline = get_pipeline()
    if not pipeline:
        return

    original_filename = uploaded_file.name
    suffix = Path(original_filename).suffix

    # Write uploaded bytes to a temp file so the parser can read it from disk
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getbuffer())
        tmp_path = tmp.name

    # Progress feedback
    progress_bar = st.progress(0, text="Starting...")
    status_text  = st.empty()

    def update_progress(step: str, percent: int):
        progress_bar.progress(percent / 100, text=step)
        status_text.caption(step)

    pipeline.progress_callback = update_progress

    try:
        # Pass both the temp path (for reading) and the original filename
        # (for naming the ChromaDB collection and in-memory chunk key)
        summary = pipeline.index(
            file_path=tmp_path,
            original_filename=original_filename,   # THE KEY FIX
        )

        # Set the active document to the original filename
        st.session_state.active_doc = original_filename
        st.session_state.chat_history = []
        pipeline.reset_conversation()

        progress_bar.empty()
        status_text.empty()

        st.success(
            f"**{original_filename}** indexed successfully.\n"
            f"- Pages processed: {summary['total_pages']}\n"
            f"- Visual pages: {summary['visual_pages']}\n"
            f"- Chunks created: {summary['total_chunks']}\n"
            f"- Processing time: {summary['processing_time_sec']}s"
        )

    except Exception as e:
        progress_bar.empty()
        status_text.empty()
        st.error(f"Indexing failed: {str(e)}")
        import traceback
        st.code(traceback.format_exc())

    finally:
        # Always clean up the temp file
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


def _show_pipeline_status():
    from config import Config
    stages = [
        ("Ingestion",   True,                      "PDF/DOCX/Excel/CSV/IMG"),
        ("VLM",         bool(Config.get_api_key()), "Qwen2.5-VL-7B"),
        ("Chunker",     True,                      "LangChain Recursive"),
        ("Embedder",    True,                      "all-MiniLM-L6-v2"),
        ("Retriever",   True,                      "BM25 + Dense + Reranker"),
        ("LLM",         bool(Config.get_api_key()), Config.get_llm_model()[:30]),
    ]
    for name, active, detail in stages:
        status = "Available" if active else "Unavailable"
        st.markdown(
            f"<small><strong>{name}</strong> — {detail} | {status}</small>",
            unsafe_allow_html=True,
        )



# Main Content Area

def render_main():
    st.markdown("""
    <h1 style='color:#E2E8F0; margin-bottom:4px;'>
        AI Document Intelligence System
    </h1>
    <p style='color:#64748B; font-size:14px; margin-top:0;'>
        Upload a document, ask questions, and review grounded answers.
    </p>
    """, unsafe_allow_html=True)

    if not st.session_state.active_doc:
        _show_welcome_screen()
        return

    # Active document banner
    st.markdown(
        f"<div class='info-card'>Current document: "
        f"<strong style='color:#93C5FD;'>{st.session_state.active_doc}</strong>"
        f"</div>",
        unsafe_allow_html=True,
    )

    # Tabs
    tab_qa, tab_extract, tab_anomaly, tab_pipeline = st.tabs([
        "Q&A",
        "Extract",
        "Anomaly Review",
        "Pipeline View",
    ])

    with tab_qa:
        render_qa_tab()
    with tab_extract:
        render_extract_tab()
    with tab_anomaly:
        render_anomaly_tab()
    with tab_pipeline:
        render_pipeline_tab()


def _show_welcome_screen():
    st.markdown("""
    <div style='text-align:center; padding:60px 20px;'>
        <h2 style='color:#94A3B8;'>Upload a document to get started</h2>
        <p style='color:#64748B; max-width:500px; margin:0 auto;'>
            Use the sidebar to upload a PDF, Word document, Excel file, CSV,
            TXT, or image. The AI will index it and you can start asking questions.
        </p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("#### Example questions you can ask:")
    cols = st.columns(3)
    examples = [
        ("Contracts", [
            "What are the termination clauses?",
            "Who are the parties involved?",
            "What is the payment schedule?",
        ]),
        ("Invoices", [
            "What is the total amount due?",
            "Are there any unusual payments?",
            "List all invoice dates and amounts",
        ]),
        ("Reports", [
            "Summarize the key findings",
            "What are the main risks mentioned?",
            "What recommendations are made?",
        ]),
    ]
    for col, (title, qs) in zip(cols, examples):
        with col:
            st.markdown(f"**{title}**")
            for q in qs:
                st.caption(f"• {q}")


# Q&A Tab

def render_qa_tab():
    # Display conversation history
    if not st.session_state.chat_history:
        st.markdown(
            "<div style='color:#64748B; text-align:center; padding:30px;'>"
            "Ask your first question about the document below."
            "</div>",
            unsafe_allow_html=True,
        )
    else:
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                st.markdown(
                    f"<div class='chat-user'><strong>You</strong><br><br>{msg['content']}</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"<div class='chat-assistant'>"
                    f"<strong style='color:#93C5FD;'>Assistant</strong><br><br>"
                    f"{msg['content']}"
                    f"</div>",
                    unsafe_allow_html=True,
                )
                _render_answer_quality(msg)
                if msg.get("sources"):
                    _render_sources(msg["sources"])

    st.markdown("---")

    # Question input
    col_input, col_btn = st.columns([5, 1])
    with col_input:
        user_question = st.text_input(
            "Ask a question",
            placeholder="e.g. What are the payment terms?",
            label_visibility="collapsed",
            key="qa_input",
        )
    with col_btn:
        ask_btn = st.button("Ask", type="primary", use_container_width=True)

    # Quick action buttons
    st.markdown(
        "<small style='color:#64748B;'>Quick questions:</small>",
        unsafe_allow_html=True,
    )
    quick_cols = st.columns(4)
    quick_questions = [
        "Summarize this document",
        "What are the key entities?",
        "List important dates",
        "What are the main topics?",
    ]
    quick_clicked = None

    for i, (col, q) in enumerate(zip(quick_cols, quick_questions)):
        with col:
            if st.button(q, use_container_width=True, key=f"quick_{i}"):
                quick_clicked = q

    question_to_ask = (user_question if ask_btn and user_question else quick_clicked)
    if question_to_ask:
        _run_qa(question_to_ask, mode="qa")
        st.rerun()

    # Clear chat
    if st.session_state.chat_history:
        if st.button("Clear Chat", key="clear_chat"):
            st.session_state.chat_history = []
            pipeline = get_pipeline()
            if pipeline:
                pipeline.reset_conversation()
            st.rerun()


def _run_qa(question: str, mode: str = "qa"):
    """Send a question through the pipeline and add result to chat history."""
    pipeline = get_pipeline()
    if not pipeline:
        return

    # Add user message
    st.session_state.chat_history.append({
        "role":    "user",
        "content": question,
        "sources": [],
    })

    with st.spinner("Searching and generating answer..."):
        result = pipeline.query(
            question=question,
            collection_name=st.session_state.active_doc,
            mode=mode,
        )

    # Add assistant reply
    st.session_state.chat_history.append({
        "role":      "assistant",
        "content":   result["answer"],
        "sources":   result.get("sources",   []),
        "model":     result.get("model"),
        "quality":   result.get("quality", {}),
        "entities":  result.get("entities",  {}),
        "anomalies": result.get("anomalies", []),
        "retrieval": result.get("retrieval", {}),
    })


def _render_answer_quality(message: dict):
    """Show a compact confidence and faithfulness summary below an answer."""
    quality = message.get("quality") or {}
    confidence = quality.get("confidence")
    faithfulness = quality.get("faithfulness")
    model = message.get("model")

    chips = []
    if confidence is not None:
        chips.append(f"Confidence: {confidence:.2f}")
    if faithfulness is not None:
        chips.append(f"Faithfulness: {faithfulness:.2f}")
    if model:
        chips.append(f"Model: {model}")

    if chips:
        st.caption(" | ".join(chips))


def _render_sources(sources: list):
    if not sources:
        return
    html = "<div style='margin:4px 0;'><small style='color:#64748B;'>Sources: </small>"
    for s in sources:
        label = f"{s['file']} p.{s['page']}"
        if s.get("section"):
            label += f" — {s['section'][:30]}"
        html += f"<span class='source-tag'>{label}</span>"
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)



# Extract Tab

def render_extract_tab():
    st.markdown("#### Extract Structured Information")
    st.markdown(
        "<small style='color:#64748B;'>Ask for specific structured data — "
        "entities, tables, lists, summaries.</small>",
        unsafe_allow_html=True,
    )

    templates = {
        "All entities (dates, names, amounts)": (
            "Extract all named entities from this document: "
            "dates, organizations, people, money amounts, and any key terms."
        ),
        "Key clauses (contracts)": (
            "List all key clauses in this document with their clause numbers "
            "and brief descriptions."
        ),
        "Table of contents": (
            "List all sections and headings found in this document."
        ),
        "Important numbers & figures": (
            "Extract all important numerical data, statistics, and financial figures."
        ),
        "Action items / deadlines": (
            "List all action items, deadlines, and obligations mentioned."
        ),
        "Custom": "",
    }

    choice = st.selectbox("Choose extraction template", list(templates.keys()))
    if choice == "Custom":
        query = st.text_area(
            "What to extract?",
            placeholder="e.g. Extract all clauses related to liability...",
        )
    else:
        query = templates[choice]
        st.text_area("Query (editable)", value=query, key="extract_display")

    if st.button("Extract", type="primary"):
        if query:
            _run_qa(query, mode="extract")
            st.rerun()

    # Show last result
    if st.session_state.chat_history:
        last = next(
            (m for m in reversed(st.session_state.chat_history)
             if m["role"] == "assistant"),
            None,
        )
        if last:
            st.markdown("---")
            st.markdown("#### Extracted Information")
            st.markdown(
                f"<div class='answer-box'>{last['content']}</div>",
                unsafe_allow_html=True,
            )
            _render_answer_quality(last)
            _render_sources(last.get("sources", []))

            if last.get("entities"):
                st.markdown("#### Detected Entities")
                ents = last["entities"]
                cols = st.columns(min(len(ents), 3))
                for i, (cat, items) in enumerate(ents.items()):
                    with cols[i % 3]:
                        st.markdown(f"**{cat.replace('_', ' ').title()}**")
                        for item in items[:10]:
                            st.markdown(f"- {item}")



# Anomaly Tab

def render_anomaly_tab():
    st.markdown("#### Anomaly Detection")
    st.markdown(
        "<small style='color:#64748B;'>Best for invoices, "
        "financial statements, and tabular data.</small>",
        unsafe_allow_html=True,
    )
    st.info(
        "This mode analyzes your document for unusual patterns: "
        "outlier amounts, duplicates, missing data, inconsistent values."
    )

    if st.button("Scan for Anomalies", type="primary"):
        query = (
            "Analyze all numerical data, dates, and entries in this document. "
            "Flag anything that looks unusual: duplicate values, outlier amounts, "
            "missing fields, or inconsistent patterns. Be specific about what is "
            "unusual and why."
        )
        _run_qa(query, mode="anomaly")
        st.rerun()

    if st.session_state.chat_history:
        last = next(
            (m for m in reversed(st.session_state.chat_history)
             if m["role"] == "assistant"),
            None,
        )
        if last:
            st.markdown("---")
            st.markdown("#### Analysis Result")
            st.markdown(
                f"<div class='answer-box'>{last['content']}</div>",
                unsafe_allow_html=True,
            )
            _render_answer_quality(last)
            _render_sources(last.get("sources", []))

            if last.get("anomalies"):
                st.markdown("#### Statistical Anomalies Detected")
                for a in last["anomalies"]:
                    color = "#EF4444" if a["severity"] == "HIGH" else "#F59E0B"
                    st.markdown(
                        f"<div style='background:#2A1A00; border-left:4px solid "
                        f"{color}; padding:10px; margin:6px 0; border-radius:6px;'>"
                        f"<strong style='color:{color};'>[{a['severity']}]</strong> "
                        f"Value: <code>{a['raw']}</code> — "
                        f"{a['deviation']}x std dev from mean ({a['mean']})"
                        f"</div>",
                        unsafe_allow_html=True,
                    )



# Pipeline View Tab

def render_pipeline_tab():
    st.markdown("#### How This Pipeline Works")

    stages = [
        {
            "num": "01",
            "name": "Smart Ingestion",
            "desc": "Load any document format",
            "detail": "PDF / DOCX / Excel / CSV / TXT / PNG / JPG",
            "color": "#3B82F6",
        },
        {
            "num": "02",
            "name": "VLM Processing",
            "desc": "Understand visual content",
            "detail": "Qwen2.5-VL-7B extracts text from charts, tables, diagrams, and scans",
            "color": "#8B5CF6",
        },
        {
            "num": "03",
            "name": "LangChain Chunking",
            "desc": "Split text intelligently",
            "detail": "RecursiveCharacterTextSplitter respects paragraphs, sentences, and words",
            "color": "#EC4899",
        },
        {
            "num": "04",
            "name": "Embedding & Indexing",
            "desc": "Convert text to vectors",
            "detail": "all-MiniLM-L6-v2 -> 384-dim embeddings -> ChromaDB (local persistent)",
            "color": "#10B981",
        },
        {
            "num": "05",
            "name": "Hybrid Retrieval",
            "desc": "3-stage retrieval system",
            "detail": "BM25 (keywords) + Dense (semantics) -> merged -> Cross-encoder reranker",
            "color": "#F59E0B",
        },
        {
            "num": "06",
            "name": "LLM Generation",
            "desc": "Generate grounded answers",
            "detail": "Mistral/LLaMA via HuggingFace with conversation history for follow-ups",
            "color": "#EF4444",
        },
    ]

    for stage in stages:
        col_num, col_content = st.columns([1, 8])
        with col_num:
            st.markdown(
                f"<div style='background:{stage['color']}22; border:1px solid "
                f"{stage['color']}; border-radius:8px; padding:12px; text-align:center;'>"
                f"<div style='font-size:24px; color:{stage['color']}; font-weight:700;'>"
                f"{stage['num']}</div>"
                f"<div style='font-size:10px; color:#64748B; letter-spacing:0.08em;'>"
                f"STEP</div>"
                f"</div>",
                unsafe_allow_html=True,
            )
        with col_content:
            st.markdown(
                f"<div class='info-card'>"
                f"<strong style='color:{stage['color']};'>{stage['name']}</strong> "
                f"- {stage['desc']}<br>"
                f"<small style='color:#64748B;'>{stage['detail']}</small>"
                f"</div>",
                unsafe_allow_html=True,
            )

    st.markdown("---")

    # Document stats
    pipeline = get_pipeline()
    if pipeline and st.session_state.active_doc:
        info = pipeline.indexed_documents.get(st.session_state.active_doc)
        if info:
            st.markdown("#### Current Document Statistics")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Pages", info.get("total_pages", 0))
            c2.metric("Visual Pages", info.get("visual_pages", 0))
            c3.metric("Chunks", info.get("total_chunks", 0))
            c4.metric("Index Time", f"{info.get('processing_time_sec', 0)}s")

    # Retrieval breakdown for last query
    if st.session_state.chat_history:
        last = next(
            (m for m in reversed(st.session_state.chat_history)
             if m["role"] == "assistant"),
            None,
        )
        if last and last.get("retrieval"):
            ret = last["retrieval"]
            st.markdown("#### Last Query Retrieval Breakdown")
            c1, c2, c3 = st.columns(3)
            c1.metric("BM25 Candidates", ret.get("bm25_count",         0))
            c2.metric("Dense Candidates", ret.get("dense_count",        0))
            c3.metric("Final (Reranked)",  ret.get("total_candidates",   0))



# Entry Point

def main():
    init_session()

    # Warn if no API key configured
    from config import Config
    if not Config.get_api_key() or Config.get_api_key() in (
        "hf_your_token_here", "sk-or-your-key-here", ""
    ):
        st.warning(
            "**No API key found in `.env`!**\n\n"
            "Add your HuggingFace token:\n"
            "```\nHF_API_KEY=hf_your_token_here\n```\n"
            "Get a free token at: https://huggingface.co/settings/tokens"
        )

    render_sidebar()
    render_main()


if __name__ == "__main__":
    main()
