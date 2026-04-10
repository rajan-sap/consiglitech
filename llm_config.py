"""
LLM Configuration — Single source of truth
============================================

Switch between providers by changing the settings below.
Every module imports from here.

Groq:       base_url = "https://api.groq.com/openai/v1", model = "llama-3.3-70b-versatile"
LM Studio:  base_url = "http://localhost:1234/v1",        model = "local-model"
OpenAI:     base_url = None (default),                    model = "gpt-4-1106-preview"
"""

import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# Helper: read secrets from Streamlit Cloud or fall back to env vars
def _get_secret(key: str, fallback: str = "") -> str:
    """Try Streamlit secrets first, then env vars, then fallback."""
    try:
        import streamlit as st
        return st.secrets.get(key, os.getenv(key, fallback))
    except Exception:
        return os.getenv(key, fallback)

# =============================================================================
# PROVIDER TOGGLE  — change this section to switch providers
# =============================================================================

# --- Gemini (Google, OpenAI-compatible endpoint) ---
LLM_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
LLM_MODEL = "gemini-2.0-flash"
LLM_API_KEY = _get_secret("GEMINI_API_KEY")

# --- LM Studio (local) — uncomment these and comment out the block above ---
# LLM_BASE_URL = "http://localhost:1234/v1"
# LLM_MODEL = "local-model"
# LLM_API_KEY = _get_secret("LM_STUDIO_API_KEY")  # set in .streamlit/secrets.toml or env


# =============================================================================
# CONTEXT LIMITS  — prevent prompt overflow for local models
# =============================================================================

# Groq models support 128k context, so we can be more generous.
# For local models with 4096 tokens, reduce to 10000.
MAX_CONTEXT_CHARS = 30000

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
    timeout=60.0,  # Groq is fast; 60s is plenty
)
