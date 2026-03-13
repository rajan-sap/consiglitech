"""
LLM Configuration — Single source of truth
============================================

Switch between OpenAI API and LM Studio (or any OpenAI-compatible server)
by changing the settings below. Every module imports from here.

LM Studio:  base_url = "http://localhost:1234/v1", model = "local-model"
OpenAI:     base_url = None (default),             model = "gpt-4-1106-preview"
"""

import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# =============================================================================
# PROVIDER TOGGLE  — change this section to switch providers
# =============================================================================

# --- LM Studio (local) ---
LLM_BASE_URL = "http://localhost:1234/v1"
LLM_MODEL = "local-model"            # LM Studio ignores this; it serves whatever model is loaded
LLM_API_KEY = "sk-lm-lljhYQvo:LFEYMnJ2XMGJnswmNIMH"            # Set to your key if LM Studio auth is enabled

# --- OpenAI (cloud) — uncomment these and comment out the block above ---
# LLM_BASE_URL = None
# LLM_MODEL = "gpt-4-1106-preview"
# LLM_API_KEY = os.getenv("OPENAI_API_KEY")

# =============================================================================
# CONTEXT LIMITS  — prevent prompt overflow for local models
# =============================================================================

# Max characters for retrieved context (~3.5 chars per token).
# For a 4096 context window, reserve ~1000 tokens for system+question+response.
# That leaves ~3000 tokens for context ≈ 10500 chars.
MAX_CONTEXT_CHARS = 10000

def truncate_context(text: str, max_chars: int = MAX_CONTEXT_CHARS) -> str:
    """Truncate context to fit within the model's context window."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n\n[... context truncated to fit model limit ...]"

# =============================================================================
# SHARED CLIENT — import `llm_client` from this module everywhere
# =============================================================================

llm_client = OpenAI(
    base_url=LLM_BASE_URL,
    api_key=LLM_API_KEY or "not-needed",
    timeout=120.0,  # 2 minutes per request (local models are slower)
)
