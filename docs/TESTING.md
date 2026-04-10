# Testing Guide — DocIntel

## Why Test?

Imagine you build a house. Before anyone moves in, you check:
- Do the doors open and close?
- Does the plumbing work?
- Does the electricity reach every room?

Software tests do the same thing. They automatically check that every part of your app works correctly. Without tests:
- You change one thing, and something else breaks silently
- You deploy to production and users find the bugs
- You can't refactor with confidence

With tests:
- You catch bugs before users do
- You can change code and instantly know if you broke something
- You prove to employers (and yourself) that the code actually works

---

## How to Run Tests

```bash
# Run all unit tests (fast, no API key needed)
pytest tests/unit -v

# Run a specific test file
pytest tests/unit/test_upload_processor.py -v

# Run a specific test class
pytest tests/unit/test_session_retriever.py::TestSearch -v

# Run a single test
pytest tests/unit/test_generator.py::TestIsGeneralQuery::test_returns_true_for_general -v

# Run integration tests (needs GEMINI_API_KEY + ChromaDB on disk)
pytest tests/integration -v -m integration
```

---

## Test Structure

```
tests/
├── conftest.py                      # Shared test data (fixtures)
├── unit/                            # Fast tests, no external services needed
│   ├── test_metadata_extraction.py  # 15 tests
│   ├── test_generator.py            # 12 tests
│   ├── test_session_retriever.py    # 16 tests
│   ├── test_upload_processor.py     # 12 tests
│   ├── test_eval_metrics.py         # 11 tests
│   ├── test_llm_config.py           #  7 tests
│   ├── test_ingestion_config.py     #  7 tests
│   ├── test_data_counting.py        #  4 tests
│   └── test_query_limit.py          #  2 tests
└── integration/                     # Slow tests, need API key + database
    └── test_rag_pipeline.py         #  9 tests
```

**Total: 95 tests**

---

## Unit Tests vs Integration Tests

| | Unit Tests | Integration Tests |
|---|---|---|
| **Speed** | Fast (seconds) | Slow (minutes) |
| **Dependencies** | None — everything is mocked | Needs real API key + ChromaDB |
| **What they test** | One function in isolation | The whole pipeline end-to-end |
| **When to run** | Every time you change code | Before deploying |
| **Analogy** | Testing if a single brick is solid | Testing if the whole wall stands |

---

## Test-by-Test Explanation

### conftest.py — Shared Test Data

This file provides **fixtures** — reusable test data that multiple test files share.

| Fixture | What it provides | Used by |
|---|---|---|
| `sample_search_results` | 3 fake search results (BMW, Ford, Tesla) that look exactly like real retriever output | Generator tests |
| `general_queries` | 8 queries like "Hello", "Who are you?" that should NOT trigger document search | Integration tests |
| `document_queries` | 5 queries like "Tesla revenue 2023" that SHOULD trigger document search | Integration tests |

**Why?** Instead of copy-pasting test data into every file, we define it once and pytest automatically injects it wherever needed.

---

### test_metadata_extraction.py — Can We Parse User Queries?

**What it tests:** When a user asks "What was BMW's revenue in 2023?", can our code correctly extract:
- Company = BMW
- Year = 2023
- Document type = Annual Report

**15 tests covering:**

| Test | What it checks | Why it matters |
|---|---|---|
| `test_detects_bmw` | "BMW's revenue" → company = BMW | Must find the right company |
| `test_detects_ford` | "Ford's annual report" → company = Ford | Works for all companies |
| `test_detects_tesla` | "Tesla vehicle deliveries" → company = Tesla | Works for all companies |
| `test_case_insensitive_company` | "bmw revenue" (lowercase) → company = BMW | Users don't always capitalize |
| `test_no_company` | "What is machine learning?" → company = None | Don't hallucinate a company |
| `test_extracts_year` | "BMW revenue in 2023" → year = 2023 | Must find the year |
| `test_extracts_first_year` | "Compare 2021 and 2022" → year = one of them | Multiple years in query |
| `test_no_year` | "Overall BMW strategy" → year = None | Don't invent a year |
| `test_rejects_non_year_numbers` | "Page 1234" → year = None | 1234 is not a valid year |
| `test_detects_annual_report` | "annual report highlights" → Annual Report | Correct doc type |
| `test_detects_news_article` | "news article about Ford" → News Article | Correct doc type |
| `test_no_doc_type` | "Tesla revenue?" → None | Don't guess doc type |
| `test_all_fields_present` | "Ford's annual report for 2022" → all 3 fields | Everything works together |
| `test_empty_query` | "" → all None | Empty input doesn't crash |
| `test_special_characters` | "BMW's/Ford's EBIT (2023)?" → doesn't crash | Weird punctuation is safe |

**Real-world impact:** If metadata extraction breaks, the retriever searches the wrong documents. A user asks about Tesla 2023, but gets BMW 2021 results. Bad answers, bad experience.

---

### test_generator.py — Does the Brain of the App Work?

**What it tests:** The generator is the brain — it decides whether to search documents or just chat, then produces the answer. We test it by **mocking** the LLM (faking its responses) so we don't need an API key.

**12 tests in 3 groups:**

#### Group 1: Intent Classification (4 tests)
*"Should we search documents or just chat?"*

| Test | What it checks |
|---|---|
| `test_returns_true_for_general` | LLM says "GENERAL" → skip document search |
| `test_returns_false_for_document` | LLM says "DOCUMENT" → do document search |
| `test_defaults_to_false_on_exception` | LLM crashes → default to document search (safer) |
| `test_rejects_unexpected_label` | LLM says "MAYBE" → treat as document (safe default) |

**Why the safe default?** If we wrongly classify a document question as general, the user gets a made-up answer instead of one grounded in real data. That's worse than unnecessarily searching documents for a greeting.

#### Group 2: Knowledge Base Generation (5 tests)
*"Does the full RAG pipeline produce answers?"*

| Test | What it checks |
|---|---|
| `test_document_route_calls_retriever` | Document questions actually search the database |
| `test_general_route_skips_retrieval` | General questions don't waste time searching |
| `test_general_return_details_has_empty_context` | General answers have no document context (as expected) |
| `test_unavailable_retriever_returns_message` | If ChromaDB is down, user gets a clear error, not a crash |
| `test_empty_context_returns_helpful_message` | If no documents match, user gets guidance, not silence |

#### Group 3: Upload Generation (3 tests)
*"Does chat-with-your-documents work?"*

| Test | What it checks |
|---|---|
| `test_returns_answer_from_uploaded_docs` | Searches uploaded docs and generates an answer |
| `test_no_results_returns_helpful_message` | If nothing matches, tells user to try again |
| `test_uses_k5_for_search` | Always retrieves top 5 results (not 3, not 10) |

---

### test_upload_processor.py — Can We Read User Files?

**What it tests:** When a user uploads a PDF or DOCX, can we extract the text, split it into chunks, and attach the right metadata?

**12 tests in 4 groups:**

#### Group 1: PDF Parsing (3 tests)

| Test | What it checks |
|---|---|
| `test_extracts_text_from_pages` | A 2-page PDF produces 2 documents with correct text |
| `test_page_numbers_are_sequential` | Page numbers are [1, 2], not [0, 1] or random |
| `test_empty_pdf_returns_empty_list` | A blank PDF returns [] instead of crashing |

**How we test without real PDFs:** We create tiny PDFs in memory using the `fpdf2` library. No files on disk needed.

#### Group 2: DOCX Parsing (2 tests)

| Test | What it checks |
|---|---|
| `test_extracts_paragraphs` | A DOCX with 2 paragraphs produces text containing both |
| `test_empty_docx_returns_empty_list` | An empty DOCX returns [] instead of crashing |

#### Group 3: Chunking (3 tests)

| Test | What it checks |
|---|---|
| `test_chunks_have_correct_metadata` | Every chunk has `file_name`, `page_number`, `document_type: "User Upload"` |
| `test_long_text_produces_multiple_chunks` | 8000 chars of text → multiple chunks (not one giant chunk) |
| `test_empty_input_returns_empty` | No documents in → no chunks out |

**Why chunking matters:** LLMs have token limits. A 200-page PDF can't be sent whole. We split it into ~1000-char chunks, embed each one, and only retrieve the most relevant chunks for each question.

#### Group 4: File Routing (4 tests)

| Test | What it checks |
|---|---|
| `test_pdf_routing` | `.pdf` files go through the PDF parser |
| `test_docx_routing` | `.docx` files go through the DOCX parser |
| `test_unsupported_extension_returns_empty` | `.png` files return [] (not supported) |
| `test_no_extension_returns_empty` | Files without extensions return [] |

---

### test_session_retriever.py — Does the In-Memory Database Work?

**What it tests:** The session retriever is a temporary database that lives only while the user's browser tab is open. It stores uploaded document embeddings and searches them.

**16 tests in 4 groups:**

#### Group 1: Initialization (2 tests)

| Test | What it checks |
|---|---|
| `test_creates_without_error` | Can we create an empty database? |
| `test_starts_empty` | New database has 0 documents and no file names |

#### Group 2: Adding Documents (6 tests)

| Test | What it checks |
|---|---|
| `test_returns_chunk_count` | Adding 3 chunks returns 3 |
| `test_empty_list_returns_zero` | Adding nothing returns 0 |
| `test_tracks_file_name` | After adding "report.pdf", it appears in the file list |
| `test_no_duplicate_file_names` | Adding the same file twice doesn't duplicate its name |
| `test_multiple_files_tracked` | Adding file_a and file_b tracks both |
| `test_doc_count_increases` | Count goes from 0 to 3 after adding 3 chunks |

#### Group 3: Searching (4 tests)

| Test | What it checks |
|---|---|
| `test_returns_results_after_adding` | Search finds documents after they're added |
| `test_result_format` | Each result has `document`, `metadata`, `cosine_similarity` |
| `test_empty_store_returns_empty` | Searching an empty database returns [] |
| `test_relevant_result_ranks_higher` | "vehicle deliveries" ranks the vehicle chunk first |

**The relevance test is key:** It proves that vector similarity actually works — similar meaning = higher score.

#### Group 4: Clearing (4 tests)

| Test | What it checks |
|---|---|
| `test_clear_resets_count` | After clear, count is 0 |
| `test_clear_resets_file_names` | After clear, file list is empty |
| `test_search_after_clear_returns_empty` | Can't find old documents after clearing |
| `test_can_add_after_clear` | Can add new documents after clearing (database isn't broken) |

---

### test_eval_metrics.py — Are Our Measurement Tools Correct?

**What it tests:** Before measuring how good our RAG is, we need to make sure our measurement tools themselves are correct.

**11 tests covering:**

| Test | What it measures | Analogy |
|---|---|---|
| `test_perfect_match_rank_1` | MRR = 1.0 when answer is first result | Googling and the answer is result #1 |
| `test_match_rank_2` | MRR = 0.5 when answer is second result | Answer is result #2 — decent but not great |
| `test_no_match` | MRR = 0.0 when answer isn't found | Google returned nothing useful |
| `test_hit_at_k1` | Hit Rate @1 = 1.0 when top result is correct | First result is what you need |
| `test_miss_at_k1_hit_at_k3` | Hit Rate @1 = 0, @3 = 1.0 | Answer exists but not in first result |
| `test_mrr_dataset_average` | Average MRR across multiple queries | Overall system quality |
| `test_hit_rate_dataset` | Average Hit Rate across dataset | How often do we find the answer? |
| `test_returns_expected_keys` | Latency measurement has p50 and p95 | Measuring response speed correctly |
| `test_captures_result` | Latency wrapper captures the function's return value | Timing doesn't eat the result |
| `test_extract_from_none_usage` | Handles missing token data gracefully | API response might lack usage info |
| `test_average_token_cost` | Correct averaging of token counts | Cost estimation works |

---

### test_llm_config.py — Is the LLM Wired Up Correctly?

**7 tests checking:**

| Test | What it checks | Why |
|---|---|---|
| `test_model_name_is_nonempty_string` | Model name exists | Can't call an API without a model name |
| `test_base_url_is_valid` | URL starts with "http" or is None | Bad URL = all API calls fail |
| `test_client_is_openai_instance` | The client is a real OpenAI object | Import errors would break everything |
| `test_api_key_is_not_placeholder` | Key isn't "sk-xxx" or "not-needed" | Placeholder keys = auth failures in production |
| `test_short_text_unchanged` | Short text passes through truncation untouched | Don't modify text that already fits |
| `test_long_text_truncated` | 50,000 chars gets cut to max limit | Prevents prompt overflow crashes |
| `test_exact_boundary_no_truncation` | Text exactly at the limit is not truncated | Off-by-one errors |

---

### test_ingestion_config.py — Are the Settings Sane?

**7 tests checking:**

| Test | What it checks | Why |
|---|---|---|
| `test_embedding_model_is_set` | Model name is a non-empty string | Missing model = can't create embeddings |
| `test_chunk_sizes_positive` | Chunk sizes > 0 | Zero or negative chunk size = crash |
| `test_report_chunks_larger_than_news` | Report chunks (1000) > News chunks (500) | Reports need more context per chunk |
| `test_overlap_less_than_chunk` | Overlap (200) < chunk size (1000) | Overlap >= chunk = infinite loop |
| `test_company_folders_defined` | BMW, Ford, Tesla are in the list | Missing company = documents not ingested |
| `test_supported_file_types` | .pdf is supported | Can't process PDFs if not listed |
| `test_batch_size_reasonable` | Batch size between 1 and 10,000 | 0 = no processing, 1M = memory crash |

---

### test_data_counting.py — Does the Sidebar Show Correct Numbers?

**4 tests checking:**

| Test | What it checks |
|---|---|
| `test_returns_defaults_when_dir_missing` | On Streamlit Cloud (no ./data), returns hardcoded defaults (9 docs, 3 companies) |
| `test_counts_real_files` | Locally, counts actual files in each company folder |
| `test_ignores_subdirectories` | Doesn't count folders as documents |
| `test_empty_data_dir` | Empty data directory returns zeros, not crashes |

---

### test_query_limit.py — Is the Rate Limit Configured Correctly?

**2 tests checking:**

| Test | What it checks | Why |
|---|---|---|
| `test_max_queries_is_ten` | MAX_FREE_QUERIES = 10 | Accidentally changing this drains your API budget |
| `test_limit_window_is_one_hour` | LIMIT_WINDOW_SECONDS = 3600 | Must be exactly 1 hour, not 1 minute or 1 day |

---

### test_rag_pipeline.py (Integration) — Does Everything Work Together?

**9 tests that hit the real LLM and real database:**

| Test | What it checks |
|---|---|
| `test_general_queries` | All 8 general queries are classified as GENERAL (no false document searches) |
| `test_document_queries` | All 5 document queries are classified as DOCUMENT (no missed searches) |
| `test_bmw_query_returns_bmw_chunks` | Asking about BMW returns BMW documents, not Ford or Tesla |
| `test_tesla_query_returns_tesla_chunks` | Asking about Tesla returns Tesla documents |
| `test_metadata_filter_narrows_results` | Filtering by company=Ford returns only Ford results |
| `test_factual_answer` | "Tesla revenue 2023?" produces a real answer mentioning Tesla |
| `test_general_answer` | "What can you do?" answers without citing fake financials |
| `test_comparison_mentions_both_companies` | "Compare BMW and Ford" mentions at least one of them |
| `test_out_of_scope_no_hallucination` | "Apple revenue?" doesn't fabricate Apple data |

**These are the most important tests** — they prove the entire RAG pipeline works end-to-end with real data.

---

## Testing Concepts Glossary

| Term | Meaning | Example |
|---|---|---|
| **Unit test** | Tests one small function in isolation | Does `extract_metadata_from_query` parse "BMW" correctly? |
| **Integration test** | Tests multiple components working together | Query → classify → retrieve → generate → answer |
| **Fixture** | Reusable test data | `sample_search_results` — fake retriever output |
| **Mock** | A fake replacement for a real service | Fake LLM that returns "GENERAL" without calling the API |
| **Assertion** | The check that says "this must be true" | `assert meta["company"] == "BMW"` |
| **Marker** | A label on a test | `@pytest.mark.unit` — this is a fast test |
| **Coverage** | What percentage of code is tested | We test all core functions |
| **Edge case** | Unusual input that might break things | Empty string, missing files, weird characters |
| **Regression** | A bug that appears when you change something | Fixing retriever breaks generator |
