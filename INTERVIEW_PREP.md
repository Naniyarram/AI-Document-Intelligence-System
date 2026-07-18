# Interview Preparation Guide

## Project: Document Intelligence Platform

This guide is built specifically for this project:
- Multimodal document ingestion across `PDF`, `DOCX`, `XLSX`, `CSV`, `TXT`, `PNG/JPG`, and `PPTX`
- Visual understanding using a VLM plus OCR fallback
- Hybrid retrieval using `BM25 + dense embeddings + cross-encoder reranking`
- Grounded answer generation with LLM provider abstraction
- Confidence and faithfulness signals in the UI
- Extraction and anomaly review workflows for business documents

Use this as a spoken interview cheat sheet:
- Start with the **30-60 second answer**
- Go into **deeper explanation** only if the interviewer probes
- Practice the **follow-up** and **what-if** questions out loud

---

## 1. Project Overview & Architecture

### Q1. Give me a concise overview of this project.
**30-60 second answer**

I built a Document Intelligence Platform for business documents like invoices, contracts, reports, and scanned files. The system ingests text and visual content, converts it into searchable chunks, stores embeddings in ChromaDB, retrieves relevant context using hybrid retrieval, and then generates grounded answers with source references. I also added structured extraction, anomaly review, conversation memory, and answer quality signals so it feels closer to a production workflow than a simple RAG demo.

**If probed deeper**

The pipeline has six stages: ingestion, VLM/OCR processing, chunking, embedding and indexing, hybrid retrieval, and grounded generation. I kept those stages separate because it made debugging easier and let me handle failures gracefully. For example, if the reranker is unavailable, retrieval still works. If the remote LLM is unreachable, the system can still return a grounded fallback answer instead of failing hard.

**Follow-up: Why does this matter in the real world?**

The practical value is reducing manual document review time. Finance teams can verify totals and invoice fields, legal teams can query clauses, and compliance teams can retrieve evidence from messy documents without manually scanning pages.

**What if the interviewer asks: Is this a demo or something closer to production?**

I would say it is still a project, but I intentionally made several production-minded choices: persistent indexing, provider abstraction, fallback models, local degradation paths, chunk rebuild after restart, test coverage, and Docker support.

---

### Q2. Walk me through the architecture from upload to answer.
**30-60 second answer**

When a user uploads a file, the pipeline first detects the document type and loads it into normalized page objects. Visual or scanned pages go through the VLM or OCR path, then the content is chunked using LangChain's recursive splitter. I embed the chunks with `all-MiniLM-L6-v2`, store them in ChromaDB, and keep in-memory chunks for BM25. At query time, I run BM25 and dense retrieval, merge and deduplicate the candidates, rerank them with a cross-encoder, and then pass the top context to the LLM to generate a grounded answer with sources.

**If probed deeper**

I separate indexing-time work from query-time work. Indexing is load -> VLM/OCR -> chunk -> embed -> persist. Querying is retrieve -> rerank -> generate -> optionally extract entities or anomalies. That distinction matters because it helps reason about latency, caching, and failure modes.

**Follow-up: What state is persisted and what is in memory?**

ChromaDB persists documents, metadata, and embeddings on disk. In-memory state holds chunk objects and indexed document summaries inside the Streamlit session. If the app restarts and chunk state is gone, I rebuild chunk objects from ChromaDB so BM25 still works.

**Edge case: What bug can happen around document identity?**

A subtle bug is indexing a temp upload filename but querying with the original filename. I fixed that by explicitly passing the original filename during indexing so the collection name and the UI's active document stay aligned.

---

### Q3. Why did you choose a staged pipeline instead of a single end-to-end flow?
**30-60 second answer**

I chose a staged pipeline because each stage has a different responsibility, failure profile, and scaling behavior. Ingestion, embedding, retrieval, and generation are easier to test and debug when they are separated. It also makes the system more adaptable, because I can swap providers or models without rewriting the whole application.

**If probed deeper**

The biggest advantage was failure isolation. For example, if VLM extraction fails, I can still index the rest of the document. If BM25 cannot initialize, dense retrieval still works. If the LLM provider is down, the user still gets a grounded fallback answer rather than a blank screen.

**Follow-up: Which stage was most error-prone?**

The most fragile areas were provider/model configuration and state consistency across reruns. In practice, many "LLM" issues were actually upstream issues like document identity mismatches, bad proxy settings, or retrieval returning weak context.

**What-if: If you rebuilt it as services, how would the stages map?**

I would split it into an ingestion/indexing worker, a retrieval/generation API, a metadata store, and a vector store, with async jobs for expensive indexing steps.

---

## 2. RAG / LLM Concepts

### Q4. Why did you choose hybrid retrieval instead of dense-only or keyword-only retrieval?
**30-60 second answer**

I used hybrid retrieval because document queries are a mix of lexical and semantic search problems. BM25 is strong when the question includes exact field names like invoice numbers, clause terms, or headings. Dense retrieval is better when the user paraphrases the content. Combining both improves recall, and the cross-encoder reranker improves precision before the answer reaches the LLM.

**If probed deeper**

In this project, the retrieval flow is `BM25_TOP_K=15`, `DENSE_TOP_K=15`, then rerank to `RERANKER_TOP_K=5`. That gives the generator enough relevant context without overloading it. Dense-only would miss some exact matches, and BM25-only would struggle on semantically phrased queries.

**Follow-up: Give me one example where BM25 helps more.**

A query like "What is the Order ID?" or "Find Invoice Number XATW..." is highly lexical, so BM25 is very useful.

**Follow-up: Give me one example where dense helps more.**

A query like "What does this document say about returning the product?" might not match the exact phrasing in the document, so dense retrieval helps bridge that semantic gap.

**Common trap to avoid in interview**

Do not just say "hybrid is best of both worlds." Explain the query patterns that justify it.

---

### Q5. How do you keep the LLM grounded in the document instead of hallucinating?
**30-60 second answer**

I constrain generation in three ways: retrieval quality, prompt design, and answer formatting. The LLM only receives retrieved document context, the system prompt explicitly tells it to answer strictly from that context, and the answer must cite sources. I also run the model at low temperature because this is factual QA, not creative generation.

**If probed deeper**

I do not treat prompt design as enough by itself. Grounding mainly depends on whether retrieval surfaces the right chunks. That is why I invested in hybrid retrieval, reranking, and preserving structured content like tables. I also expose source references and heuristic faithfulness signals in the UI so users can sanity-check answers.

**Follow-up: What would you do if hallucinations still appeared?**

I would first inspect retrieval output, not the prompt. Then I would improve chunking, retrieval filtering, and source verification. If needed, I would add stricter answer validation, citation consistency checks, or a second-pass verifier.

**Edge case: When is a fallback answer risky?**

A local extractive fallback is useful for resilience, but I would be careful using it in regulated workflows because users may assume the output quality is equivalent to the main LLM path when it is not.

---

### Q6. How do follow-up questions work in your system, and what risks come with that?
**30-60 second answer**

I maintain a bounded conversation history inside the LLM handler so users can ask follow-up questions like "What about the penalties?" or "Who is the seller?" after an earlier question. That improves usability, but it also introduces the risk of context leakage if the user changes documents or if old conversation state is not cleared properly.

**If probed deeper**

The history is capped to the last 20 messages, and I explicitly reset conversation history when switching documents. That matters because otherwise pronouns and references from one document could contaminate the next answer.

**Follow-up: Why not use full conversation history indefinitely?**

Long histories increase token usage and can actually hurt grounding because the model may latch onto stale references. For document QA, shorter, recent context is usually more reliable.

**What-if: How would you handle multi-document chat?**

I would make document context explicit in the conversation state, include document IDs in session memory, and attach retrieved evidence per turn rather than relying only on free-form history.

---

## 3. Embeddings & Vector Databases

### Q7. Why did you choose `all-MiniLM-L6-v2` and ChromaDB?
**30-60 second answer**

I chose `all-MiniLM-L6-v2` because it is lightweight, fast, local, and good enough for a project focused on practical retrieval rather than heavyweight infrastructure. I chose ChromaDB because it gives me persistent local vector storage with minimal setup, which is ideal for a local-first document intelligence project.

**If probed deeper**

The main trade-off is quality versus cost and latency. Larger embedding models can improve semantic recall, but they also increase latency, memory, and deployment complexity. For ChromaDB, the main benefit was persistence without needing to run a separate database service during development.

**Follow-up: When would you replace this embedding model?**

I would replace it if retrieval quality became a bottleneck on more complex or domain-specific corpora, especially when semantic nuance matters more than fast local inference.

**What-if: Why not use a managed vector DB?**

For a local project, a managed vector DB would add operational overhead and cost early on. In production, I would re-evaluate based on scale, multi-tenancy, filtering needs, and operational SLAs.

---

### Q8. Why did you bypass ChromaDB's built-in embedding function?
**30-60 second answer**

I bypassed Chroma's built-in embedding function because it triggered ONNX runtime issues on Windows in this setup. Instead, I compute embeddings myself with SentenceTransformers and pass the vectors directly to ChromaDB. That gave me more control over the embedding model and removed a brittle dependency path.

**If probed deeper**

This was not just a workaround; it was actually a cleaner design for this project. It decouples the vector store from the embedding model, which makes the system easier to debug and easier to swap models in the future.

**Follow-up: What trade-off does that introduce?**

I take on explicit responsibility for embedding generation, batching, and model management rather than delegating it to the DB layer. That is more code, but it gives me clearer control.

**Edge case: What if embeddings cannot be downloaded?**

I added a deterministic local hash-based embedding fallback so indexing and retrieval still work, though with weaker semantic quality.

---

### Q9. How do you manage persistence and document identity in the vector store?
**30-60 second answer**

Each document gets its own Chroma collection based on a cleaned version of the original filename, and chunk metadata stores source file, page number, section title, chunk index, and content type. That lets me reconstruct chunk objects later and show accurate sources in the UI.

**If probed deeper**

I also normalize collection names to satisfy Chroma naming rules. The important design point is keeping document identity stable across upload, indexing, querying, and restart recovery. If that mapping breaks, retrieval appears to fail even if indexing technically succeeded.

**Follow-up: How do you prevent duplicates when re-uploading the same file?**

Before creating a new collection, I delete the old one if it exists. That keeps the document index fresh instead of silently accumulating duplicate chunks.

**What-if: What would you change for enterprise multi-document corpora?**

I would likely move from one collection per file toward a metadata-driven index design where document ID, tenant, and access filters become first-class fields.

---

## 4. Data Processing & Chunking

### Q10. Why did you use LangChain's `RecursiveCharacterTextSplitter` with the current chunk settings?
**30-60 second answer**

I used recursive chunking because it tries to preserve natural boundaries like paragraphs and sentences instead of cutting text blindly at fixed positions. My defaults are `CHUNK_SIZE=400` and `CHUNK_OVERLAP=60`, which balance retrieval granularity with enough local context to answer factual questions.

**If probed deeper**

Internally I map those token-oriented targets to character-based chunking because the splitter works in characters. The overlap helps prevent losing context at chunk boundaries, especially around clauses, totals, and adjacent field labels.

**Follow-up: What happens if chunks are too small?**

Recall may improve slightly, but context becomes fragmented and the generator may need multiple chunks to answer a simple question.

**Follow-up: What happens if chunks are too large?**

Retrieval gets noisier, reranking becomes less precise, and irrelevant content can pollute the LLM context window.

**Edge case: Would you use the same chunk settings for legal contracts and spreadsheets?**

No. Contracts and tables often need different strategies. In this project I already keep tables and spreadsheet batches intact because chunking them like prose hurts usability.

---

### Q11. Why do you treat tables and spreadsheet rows differently from regular text?
**30-60 second answer**

Tables carry meaning through row and column relationships, so splitting them like paragraphs destroys important structure. I keep table pages and spreadsheet-row batches as single retrieval units so amounts, headers, and corresponding values stay together.

**If probed deeper**

This matters a lot for invoices and tabular documents. If you split a row away from its header or split a total away from the adjacent label, retrieval might still bring back a chunk, but the generator will not have enough structure to answer accurately.

**Follow-up: What about very large tables?**

For very large tables, I would use table-aware segmentation, like chunking by row groups while repeating header context in each chunk.

**What-if: Why not parse tables into structured JSON first?**

That can be a good next step, but it raises complexity. For this version, preserving rows as readable searchable text was a pragmatic midpoint that still works well for retrieval.

---

### Q12. How does the ingestion layer decide whether something is text, scanned, image-heavy, or tabular?
**30-60 second answer**

The ingestion layer uses file-type-specific parsing plus simple heuristics. For PDFs, it checks whether a page has text, embedded images, or neither. If a page has no text but has an image, it is treated as scanned. If it has text but also looks very image-heavy, it may be treated as an image page. It also uses table-like heuristics such as pipe and tab patterns to classify tabular text.

**If probed deeper**

This is intentionally pragmatic rather than perfect. The goal is to route the content to the most useful downstream path. It is not a full document layout engine.

**Follow-up: Where can this break?**

It can struggle on mixed-layout PDFs, low-text chart pages, or tables that do not use obvious delimiters. It can also misclassify visually dense pages with short text.

**What-if: How would you improve it?**

I would add layout-aware parsing, visual classifiers, and stronger table detection rather than relying only on heuristics.

---

## 5. Prompt Engineering

### Q13. Explain your prompt design for grounded QA.
**30-60 second answer**

My system prompt is strict and task-oriented. It tells the model to answer only from the provided context, avoid external knowledge, preserve exact numbers, and cite sources. I pair that with a user message that includes retrieved chunks labeled by source and page, so the model sees both the evidence and the task framing clearly.

**If probed deeper**

I also vary the instruction by mode. QA, extraction, summarization, and anomaly review are different tasks, so they should not share exactly the same prompt. For example, extraction asks for structured output, while anomaly review asks the model to reason about unusual patterns.

**Follow-up: Why do you keep temperature low?**

Because this is a factual document task. I want the model to be deterministic and evidence-driven rather than creatively rephrasing beyond the context.

**What-if: What is the biggest limitation of prompt engineering here?**

Prompting cannot compensate for weak retrieval. If the wrong chunks come in, the model can still produce a fluent but misleading answer.

---

### Q14. How did you handle broad queries like "Explain this document"?
**30-60 second answer**

I found that overview-style questions needed special handling because a generic QA prompt often returned raw snippets instead of a clean summary. So I detect broad overview queries and switch to a summary-oriented instruction that asks for document type, parties, dates, totals, items, and notable notes in short bullet points.

**If probed deeper**

I also improved the local fallback path so overview queries produce a structured summary instead of just the top overlapping sentences. That matters because users often ask broad questions first, and poor first impressions can make the whole system feel unreliable.

**Follow-up: Why not use the same prompt for all questions?**

Because "what is this document?" is a different task from "what is the invoice number?" The ideal output shape is different, and retrieval context must be interpreted differently.

**What-if: How would you generalize beyond invoices?**

I would add document-type detection and more type-specific summary templates for contracts, reports, and forms.

---

### Q15. How do you prompt the VLM for charts, scanned pages, and table images?
**30-60 second answer**

I use content-type-specific prompts. For chart or image pages, I ask for all visible information, including titles, axes, values, and relationships. For scanned pages, I ask for exact text extraction while preserving structure. For tables, I explicitly ask for row-by-row output with column separators so the result is searchable and indexable later.

**If probed deeper**

That specificity is important because VLMs can otherwise summarize too aggressively. My goal is not just to describe the image, but to turn it into high-recall text that works well in downstream retrieval.

**Follow-up: What is the trade-off?**

More exhaustive prompts improve recall but can increase latency and produce noisy text. It is a balance between completeness and retrieval quality.

**What-if: What would you change for low-quality screenshots?**

I would consider image preprocessing, OCR-first pipelines for dense text, or stronger layout-aware multimodal models.

---

## 6. System Design & Scalability

### Q16. How would you productionize this beyond Streamlit?
**30-60 second answer**

I would separate the product into at least two backend paths: an async indexing service and a low-latency query service. Streamlit is fine for a demo UI, but in production I would put a proper API layer in front of retrieval and generation, move indexing into background workers, and add persistent metadata storage, observability, and authentication.

**If probed deeper**

The indexing path is naturally asynchronous because parsing, VLM extraction, chunking, and embedding can take time. Query serving should be optimized separately around retrieval latency, caching, and provider calls. I would also log retrieval candidates, model choice, latency, and answer quality signals for monitoring.

**Follow-up: What storage layers would you use?**

I would keep a vector store for embeddings, a relational store for document and user metadata, and object storage for raw uploads if needed.

**What-if: What would you not change immediately?**

I would keep the retrieval logic mostly intact at first, because the bigger win comes from separating responsibilities and improving observability before replacing components.

---

### Q17. What are the first bottlenecks if this system gets real usage?
**30-60 second answer**

The first bottlenecks are likely VLM/LLM latency, synchronous indexing in the app process, and session-bound in-memory state. Dense retrieval itself is relatively manageable at this scale, but provider calls and app-level concurrency would become limiting quickly.

**If probed deeper**

Another bottleneck is that Streamlit is not ideal as the sole orchestration layer for concurrent users. I would move indexing to queues and workers, cache repeated retrievals where useful, and think about batching or provider fallback strategies for LLM/VLM calls.

**Follow-up: What would you cache?**

I would cache parsed document artifacts, embeddings for unchanged documents, and possibly frequent retrieval results or document summaries.

**What-if: Would you shard by document, tenant, or workload type?**

In production I would first separate indexing workload from query workload. After that, tenant-aware partitioning becomes important for security and performance.

---

### Q18. How would you support enterprise multi-document search with access control?
**30-60 second answer**

I would make document ID, tenant ID, and access metadata first-class fields in indexing and retrieval. Retrieval should never search across documents the user is not authorized to access. Access control has to be enforced in the backend retrieval layer, not just the UI.

**If probed deeper**

I would likely shift from one-collection-per-file thinking toward a richer metadata and filter strategy. I would also separate document storage, metadata, permissions, and vector retrieval concerns more explicitly.

**Follow-up: How would sources be displayed safely?**

I would ensure source citations only point to content the current user can access, and I would preserve page-level metadata for auditability.

**What-if: What is the biggest risk if access control is implemented poorly?**

The biggest risk is silent data leakage through retrieval itself, even if the UI looks isolated.

---

## 7. Failure Handling & Edge Cases

### Q19. What failure modes did you explicitly design for?
**30-60 second answer**

I handled failures across multiple layers: unsupported or unavailable models, provider authentication issues, dead local proxy settings, BM25 initialization problems, reranker unavailability, embedding download failures, Chroma session loss, and LLM/VLM network failures. In each case, I tried to degrade gracefully rather than crash the entire user flow.

**If probed deeper**

Examples include dense-only fallback if BM25 fails, score-based ranking if the reranker fails, local hash embeddings if SentenceTransformers cannot load, chunk rebuild from ChromaDB after restarts, and grounded local answer fallback if remote generation is unavailable.

**Follow-up: Why invest in fallbacks instead of failing fast?**

Because this is a user-facing system. Failing fast is useful for developer visibility, but for user trust I wanted the system to remain usable when possible while still signaling degraded quality honestly.

**What-if: When would you prefer to fail hard?**

In high-stakes workflows like legal review or healthcare, I would rather fail clearly than return a lower-quality fallback without strict controls.

---

### Q20. Tell me about the provider and configuration issues you solved.
**30-60 second answer**

One important improvement was abstracting Hugging Face and OpenRouter behind the same OpenAI-compatible client interface, then selecting the backend based on the token format. I also had to handle dead proxy settings, unsupported model IDs, and key misconfiguration because many "model" failures were really environment issues.

**If probed deeper**

I added backend-safe default models, fallback model lists, clearer error messages, and logic to ignore obviously broken local proxy values like `127.0.0.1:9`. That made the system much more reliable end to end.

**Follow-up: Why does provider abstraction matter here?**

It improves portability and resilience. If one provider has a model mismatch or availability issue, the architecture does not have to change.

**What-if: What complexity does this add?**

It adds model-ID compatibility issues, more config paths, and the need for clearer observability around which backend and model were actually used.

---

### Q21. How do you debug whether a bad answer came from retrieval, chunking, or generation?
**30-60 second answer**

I debug stage by stage. First I inspect the retrieved chunks and source metadata. If retrieval looks wrong, I inspect document loading, chunking, and embeddings. If retrieval looks right but the answer is wrong, then I inspect the prompt and generation behavior. I do not start by changing the prompt, because retrieval mistakes are usually the root cause in RAG systems.

**If probed deeper**

This project helps with that because the UI surfaces sources, retrieval counts, and answer quality signals. I can also inspect chunk contents in ChromaDB-backed recovery, which helps isolate whether the data itself was indexed correctly.

**Follow-up: What is an example of a bug that looks like an LLM problem but is not?**

The temp filename versus original filename mismatch is a good example. Indexing technically succeeded, but retrieval targeted the wrong collection, so the LLM never saw the correct context.

**What-if: What if retrieval looks good but the answer is still weak?**

Then I would examine prompt instructions, answer formatting requirements, and whether the retrieved chunks contain enough complete context rather than just partially relevant fragments.

---

## 8. Evaluation & Metrics

### Q22. How would you evaluate whether this project is actually good?
**30-60 second answer**

I would evaluate it at three layers: retrieval quality, answer quality, and workflow usefulness. Retrieval quality means whether the right chunks were surfaced. Answer quality means grounding, relevance, and source correctness. Workflow usefulness means whether it actually reduces review effort for finance, legal, or compliance users.

**If probed deeper**

For offline evaluation, I would create document-specific test questions with expected answers and supporting evidence. Then I would track retrieval recall, answer faithfulness, citation correctness, latency, and user-visible failure rate. I already included a RAGAs evaluator scaffold for faithfulness, answer relevance, context recall, and context precision.

**Follow-up: Why is manual spot checking not enough?**

Because it is easy to overestimate quality from a few polished examples. You need repeatable evaluation across different document types and query patterns.

**What-if: What would you measure in production?**

Latency, indexing success rate, fallback usage rate, answer abandonment, source-click behavior, and domain-specific task completion metrics.

---

### Q23. What do the confidence and faithfulness scores in the UI mean?
**30-60 second answer**

They are heuristic signals, not calibrated probabilities. In this project, faithfulness is approximated by lexical support between the answer and retrieved context, while confidence combines lexical support with average retrieval score. They are useful for user awareness, but I would not present them as rigorous model certainty.

**If probed deeper**

I added them because they help the UI communicate that answer quality is a spectrum. But I am careful about their limitations, especially for paraphrased but correct answers or when lexical overlap is high but the interpretation is still wrong.

**Follow-up: How would you improve them?**

I would replace or supplement them with stronger offline evaluation, citation validation, and model-based or benchmark-based calibration.

**What-if: When can heuristic faithfulness be misleading?**

It can underrate a valid paraphrase or overrate an answer that repeats many words from the context but still combines them incorrectly.

---

### Q24. How does the RAGAs evaluator fit into this project?
**30-60 second answer**

I included a RAGAs evaluation component as a structured way to measure faithfulness, answer relevancy, context recall, and context precision against a question-and-ground-truth set. It is not on the critical path of the app, but it is useful for benchmarking the pipeline more systematically.

**If probed deeper**

The evaluator runs the pipeline over a set of questions, collects answers and retrieved contexts, and then computes RAGAs metrics if the package is installed. I see it as a good next step toward more rigorous regression testing.

**Follow-up: Why is this important for interview discussion?**

Because it shows I thought beyond "the demo works" and into how I would prove quality over time.

**What-if: What is still missing?**

A curated domain-specific benchmark set and regular regression tracking would make this much stronger.

---

## 9. Real-world Applications

### Q25. Which real-world workflows is this project best suited for?
**30-60 second answer**

The strongest fit is document-heavy operational workflows like invoice review, contract Q&A, internal audit support, and compliance evidence retrieval. Those use cases benefit from multimodal ingestion, grounded answers, structured extraction, and anomaly review much more than generic chatbot workflows.

**If probed deeper**

Finance operations can use it to extract totals, dates, vendor details, and anomalies. Legal or contract operations can use it to locate clauses and summarize obligations. Compliance teams can use it to find supporting evidence from messy document collections.

**Follow-up: Which use case would you prioritize first?**

I would prioritize invoice and business document review because the structure is repetitive enough to deliver quick value while still showcasing multimodal and retrieval depth.

**What-if: Why not position it as a general-purpose knowledge assistant?**

Because this system is strongest when it is grounded in business documents with traceable evidence, not as an open-domain assistant.

---

### Q26. Where would you be cautious about deploying this in the real world?
**30-60 second answer**

I would be cautious in high-stakes domains where a partially correct answer is still risky, like legal final review, healthcare, or regulated financial decisions. In those contexts, fallbacks, heuristic quality scores, and imperfect OCR can create false confidence if not wrapped in stricter controls.

**If probed deeper**

Before deploying there, I would add stronger evaluation, explicit human review loops, audit trails, access control, and possibly stricter failure policies instead of best-effort degradation.

**Follow-up: Does that mean the project is not useful?**

No. It means the right deployment scope matters. It is already useful as an analyst-assist or review acceleration tool, even if it should not be the final decision-maker in high-risk settings.

**What-if: How would you communicate that to stakeholders?**

I would position it as a decision-support system with evidence retrieval, not an autonomous authority.

---

## 10. Behavioral / Project Discussion

### Q27. What was the hardest engineering problem you solved in this project?
**30-60 second answer**

One of the most instructive issues was that successful indexing could still lead to empty retrieval because the uploaded file was stored under a temp filename while the UI queried using the original filename. It looked like an LLM problem on the surface, but the root cause was document identity mismatch between indexing and querying.

**If probed deeper**

I fixed it by passing the original filename into the indexing path and using that as the collection key consistently. That bug reinforced an important lesson: in RAG systems, state consistency and document identity can matter as much as model quality.

**Follow-up: What did that teach you as an engineer?**

It taught me to debug AI systems from the data path upward instead of blaming the model first.

**What-if: What was the second-biggest issue?**

Provider connectivity and configuration issues. Broken proxy settings and unsupported models can look like model logic problems when they are really environment and infrastructure problems.

---

### Q28. If you had two more weeks, what would you improve first?
**30-60 second answer**

I would prioritize evaluation, production architecture, and stronger structured handling for tables and document types. Specifically, I would add a more robust benchmark set, move indexing into async workers, and improve document-type-specific extraction and summarization logic.

**If probed deeper**

I would not jump to replacing every model first. The biggest ROI would come from better measurement, stronger retrieval observability, and safer system boundaries. Once those are in place, model upgrades become more meaningful.

**Follow-up: What would you deliberately not do yet?**

I would not over-invest in UI redesign or model swapping before I had better evaluation and clearer production requirements.

**What-if: If the goal shifted to enterprise readiness?**

Then I would prioritize authentication, access control, metadata filtering, auditability, and API/service separation.

---

## Top 20 Must-Know Questions With Strong Answers

1. **What does this project do?**  
It is a multimodal document intelligence platform that ingests business documents, indexes them, retrieves relevant context with hybrid retrieval, and generates grounded answers with citations.

2. **Why is this more than a simple chatbot?**  
Because it has a full retrieval pipeline, visual document understanding, source-aware generation, structured extraction, anomaly review, and persistence.

3. **What are the six pipeline stages?**  
Ingestion, VLM/OCR processing, chunking, embedding/indexing, hybrid retrieval, and grounded generation.

4. **Why hybrid retrieval?**  
Because exact fields and headings need lexical retrieval, while paraphrased questions benefit from semantic retrieval; reranking then improves precision.

5. **Why keep tables intact?**  
Because splitting rows or headers destroys relationships between labels and values and hurts both retrieval and answer quality.

6. **Why use `RecursiveCharacterTextSplitter`?**  
Because it preserves natural boundaries like paragraphs and sentences better than naive fixed-size chunking.

7. **Why use `all-MiniLM-L6-v2`?**  
It is fast, local, lightweight, and good enough for a production-minded project focused on practical retrieval quality.

8. **Why ChromaDB?**  
It gives persistent local vector storage with low setup overhead, which fits a local-first project well.

9. **Why bypass Chroma's built-in embeddings?**  
To avoid ONNX runtime issues and to keep explicit control over the embedding model and vectors.

10. **How do you keep the LLM grounded?**  
By sending only retrieved document context, using strict context-only prompting, low temperature, and source citations.

11. **How do follow-up questions work?**  
The LLM handler keeps bounded conversation history, but I reset it when switching documents to avoid context leakage.

12. **What happens if BM25 or reranking fails?**  
The system degrades gracefully to dense-only retrieval or score-based ranking.

13. **What happens if the LLM provider is unavailable?**  
The app can return a grounded local fallback answer instead of failing completely.

14. **Why support both Hugging Face and OpenRouter?**  
It improves portability and resilience while keeping a common OpenAI-compatible integration layer.

15. **How do you handle visual pages?**  
Image-heavy or scanned pages go to a VLM, with OCR as a local fallback if VLM is unavailable.

16. **How do you recover after app restarts?**  
I rebuild chunk objects from ChromaDB so BM25 and retrieval still work even if in-memory session state is lost.

17. **What do confidence and faithfulness mean?**  
They are heuristic signals based on answer-context overlap and retrieval scores, not true model calibration.

18. **How would you evaluate this properly?**  
With retrieval metrics, grounded answer quality metrics, citation correctness, workflow outcomes, and RAGAs-style benchmarking.

19. **What was the hardest bug?**  
A document identity mismatch where indexing used a temp filename and querying used the original filename, causing silent retrieval failure.

20. **What would you improve next?**  
Evaluation rigor, async production architecture, richer table/document-type handling, and enterprise controls like access filtering.

---

## Strong Answer vs Weak Answer Signals

### Strong answer signals
- Explains the system in terms of **user workflow and system behavior**, not just libraries
- Clearly separates **indexing-time** and **query-time** logic
- Can justify **hybrid retrieval** with concrete examples
- Understands that **retrieval quality drives generation quality**
- Speaks honestly about **trade-offs, limitations, and fallback quality**
- Connects architecture choices to **reliability, debugging, and scale**
- Knows where the project is **production-minded** and where it is still a project

### Weak answer signals
- Describes only the UI and says "it uses RAG" without explaining how
- Over-focuses on model names and under-explains retrieval and chunking
- Claims prompt engineering alone solves hallucinations
- Cannot explain why tables, scans, and structured content are special
- Treats heuristic scores as formal evaluation
- Avoids discussing failure modes or trade-offs
- Says "it scales" without explaining architecture changes

---

## Common Mistakes Candidates Make

- Saying "I built a chatbot" instead of "I built a document intelligence system"
- Confusing OCR, VLM extraction, and LLM generation as the same thing
- Not explaining why `BM25`, dense search, and reranking each exist
- Ignoring state management and document identity issues
- Assuming ChromaDB is the embedding model rather than the vector store
- Forgetting that chunking strategy heavily impacts retrieval quality
- Giving generic answers about RAG without referencing project-specific choices
- Overstating confidence/faithfulness signals as if they are calibrated metrics
- Skipping real production concerns like access control, async indexing, or observability

---

## Final Practice Advice

- Practice every main answer in under one minute first.
- Then practice the deeper explanation without sounding memorized.
- If an interviewer pushes on trade-offs, lead with what you optimized for in this specific project: **grounded outputs, practical resilience, and business-document usability**.
- If you do not know something, anchor your answer in the system you actually built and explain how you would reason from there.
