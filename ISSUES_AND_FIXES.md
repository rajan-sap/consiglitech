# Issues & Fixes Log

A chronological record of every problem encountered during development and how it was resolved.

---

## 1. JSON Parsing Error in Synthetic QA Dataset Generation

**When:** Dataset generation with local LLM  
**File:** `evaluation/dataset_generator.py`

**Problem:** The LLM sometimes returned QA pairs as a flat dict (`{"question": "...", "answer": "..."}`) instead of a list of dicts, or wrapped the JSON in markdown code fences (`` ```json ... ``` ``). The parser expected a strict `[{...}, {...}]` format, causing `"string indices must be integers"` errors.

**Fix:** Added robust JSON extraction logic that handles three formats:
1. **List of dicts** — standard expected format, used as-is
2. **Flat dict with `q`/`a` keys** — wrapped into a single-element list
3. **Nested list-of-dicts inside any dict value** — extracted recursively
4. **Markdown fence stripping** — `re.sub` removes `` ```json ``` `` wrappers before parsing

---

## 2. OpenAI API Quota Limit (429 Error)

**When:** First evaluation run (40/80 QA pairs generated)  
**File:** `evaluation/evaluate.py`

**Problem:** OpenAI rate-limited the API at 40 out of 80 generation calls, returning HTTP 429. Evaluation pipeline crashed mid-run.

**Fix:** Switched the entire project to **LM Studio** (local OpenAI-compatible server at `http://localhost:1234/v1`). Created a centralized `llm_config.py` so every module imports from one place, making provider switching a one-line change.

**Files changed:**
- `llm_config.py` — new file, single source of truth
- `generation/generator.py` — imports from `llm_config`
- `retrieval/retriever.py` — imports from `llm_config`
- `retrieval/utils.py` — imports from `llm_config`
- `evaluation/evaluate.py` — imports from `llm_config`
- `evaluation/dataset_generator.py` — imports from `llm_config`

---

## 3. LM Studio Authentication Failure

**When:** First attempt to use LM Studio  
**File:** `llm_config.py`

**Problem:** LM Studio returned auth errors when called with the default empty API key. The server had authentication enabled.

**Fix:** Set the LM Studio API key via environment variable or `.streamlit/secrets.toml`:
```python
LLM_API_KEY = _get_secret("LM_STUDIO_API_KEY")  # reads from secrets or env
```

---

## 4. Context Window Overflow (4096-token limit)

**When:** Generation step with local model  
**File:** `llm_config.py`, `generation/generator.py`, `evaluation/evaluate.py`

**Problem:** The local model has a 4096-token context window. The RAG pipeline aggregates context from multiple retrieved chunks across decomposed sub-queries, often exceeding this limit. The model either truncated silently or produced garbage output.

**Fix:** Added a `truncate_context()` helper in `llm_config.py` with `MAX_CONTEXT_CHARS = 10000` (~3000 tokens reserved for context, leaving ~1000 tokens for system prompt + question + response). Applied it in both the generator and evaluator before passing context to the LLM.

```python
def truncate_context(text: str, max_chars: int = MAX_CONTEXT_CHARS) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n\n[... context truncated to fit model limit ...]"
```

---

## 5. Evaluation Pipeline Crashes on Individual Failures

**When:** Evaluation run with local model (timeouts on complex queries)  
**File:** `evaluation/evaluate.py`

**Problem:** When a single query timed out (>120s) during the generation loop, the entire evaluation pipeline crashed with an unhandled exception, losing all results collected so far.

**Fix:** Wrapped each generation call in `try/except`. Failed queries produce a `"[generation failed]"` placeholder answer instead of crashing the pipeline. The final metrics still compute over all 80 samples (failed ones simply score 0 on quality metrics).

```python
try:
    result = generate_answer(query, return_details=True)
    answer = result["answer"]
except Exception as e:
    answer = "[generation failed]"
```

---

## 6. RAGAS Import API Changes (v0.4.3)

**When:** Attempting to run RAGAS LLM-as-judge evaluation  
**File:** `evaluation/evaluate.py`

**Problem:** RAGAS v0.4.3 changed its import paths and metric APIs:
- `from ragas.metrics import faithfulness` → deprecated
- Metrics are now classes that need to be instantiated, not module-level constants
- `LangchainLLMWrapper` deprecated in favor of new wrappers

**Fix:** Updated imports to use `ragas.metrics.collections` and instantiate metric objects. However, since RAGAS requires ~400+ LLM-as-judge calls (impractical with a local model), set `SKIP_RAGAS = True` and rely on custom retrieval metrics (MRR, Hit Rate @k, latency, token cost) instead.

---

## 7. RAGAS Impractical with Local Models

**When:** Evaluation planning  
**File:** `evaluation/evaluate.py`

**Problem:** RAGAS evaluation (faithfulness, answer relevancy, context precision, context recall) requires the LLM to judge each of the 80 samples multiple times — roughly 400+ LLM calls. With a local model running at ~60s per call, this would take **6+ hours**.

**Fix:** Set `SKIP_RAGAS = True` in `evaluate.py`. Custom retrieval-only metrics (MRR, Hit Rate @1/3/5/10, latency percentiles, token economics) run without any LLM calls and complete in seconds. RAGAS can be re-enabled by setting `SKIP_RAGAS = False` and switching back to OpenAI in `llm_config.py`.

---

## 8. Generic Questions Routed Through RAG Pipeline

**When:** User asked "What AI model are you using?" and got an answer stuffed with irrelevant annual report context  
**File:** `generation/generator.py`

**Problem:** Every query was routed through the full RAG pipeline (decompose → retrieve documents → generate from context), so generic questions like *"What AI model are you using?"* got irrelevant annual report chunks injected as "ground truth." The model then tried to answer the meta question using Tesla/Ford/BMW filings, producing confused and unhelpful responses.

**Fix — Two-route architecture:**

| Route | Trigger | Behavior |
|-------|---------|----------|
| **General** | Heuristic regex matches greetings, meta/system questions, general knowledge | Skips retrieval entirely; LLM answers directly with a general-purpose system prompt |
| **Document** | Everything else | Full RAG pipeline with an improved system prompt that cites sources and refuses to fabricate |

Added `is_general_query()` classifier with regex patterns for:
- **Greetings:** "Hello", "Thanks", "Good morning"
- **Meta questions:** "What AI model are you using?", "Who are you?", "What can you do?"
- **General knowledge:** "Define machine learning", "What is the capital of France?"

Also improved both system prompts:
- `SYSTEM_PROMPT_RAG` — instructs the model to use ONLY retrieved context and refuse to fabricate
- `SYSTEM_PROMPT_GENERAL` — instructs the model to answer from its own knowledge without referencing documents

All 15 test cases (10 general, 5 document) classified correctly. No UI changes needed.

---

## Quick Reference: Key Configuration

| Setting | Location | Current Value |
|---------|----------|---------------|
| LLM Provider | `llm_config.py` | LM Studio (`localhost:1234`) |
| Context limit | `llm_config.py` | 10,000 chars |
| Client timeout | `llm_config.py` | 120s |
| RAGAS enabled | `evaluation/evaluate.py` | `SKIP_RAGAS = True` |
| QA pairs per report | `evaluation/config.py` | 10 (80 total) |
| Embedding model | `ingestion/constants.py` | `BAAI/bge-base-en-v1.5` |
