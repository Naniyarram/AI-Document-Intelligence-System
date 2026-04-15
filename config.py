
# config.py — Central Configuration
#-------------------------------------------------------------------------------
# Supports two API backends:
#   1. HuggingFace Inference Providers  (token starts with hf_)
#      Base URL: https://router.huggingface.co/v1
#      Models:   HuggingFace model IDs (e.g. meta-llama/...)

#   2. OpenRouter  (token starts with sk-or-)
#      Base URL: https://openrouter.ai/api/v1
#      Models:   OpenRouter model IDs (e.g. mistralai/mistral-7b-instruct:free)

# The code auto-detects which backend to use based on the token prefix.


import os
from pathlib import Path
from dotenv import load_dotenv

# Find .env file starting from this file's directory
_here     = Path(__file__).resolve().parent
_env_path = _here / ".env"
load_dotenv(dotenv_path=_env_path if _env_path.exists() else None)


def _disable_broken_local_proxy_env():
    """
    Some Windows setups inject a dead localhost proxy such as 127.0.0.1:9.
    That breaks external API calls for Hugging Face and OpenRouter.

    We only clear obviously broken local proxy values so normal proxy setups
    are left untouched.
    """
    proxy_vars = [
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "GIT_HTTP_PROXY",
        "GIT_HTTPS_PROXY",
    ]
    bad_markers = ("127.0.0.1:9", "localhost:9")

    for var_name in proxy_vars:
        value = os.getenv(var_name, "").strip().lower()
        if value and any(marker in value for marker in bad_markers):
            os.environ.pop(var_name, None)


_disable_broken_local_proxy_env()


def _clean(value: str) -> str:
    """Strip surrounding quotes that Windows users sometimes add."""
    if value:
        value = value.strip()
        if (value.startswith('"') and value.endswith('"')) or \
           (value.startswith("'") and value.endswith("'")):
            value = value[1:-1]
    return value


class Config:
    """
    Central config.  Import anywhere: from config import Config

    Auto-detects HuggingFace vs OpenRouter based on token prefix:
      hf_...     → HuggingFace Inference Providers router
      sk-or-...  → OpenRouter
    """

    # Raw token values
    HF_TOKEN:           str = _clean(os.getenv("HF_TOKEN",           ""))
    HF_API_KEY:         str = _clean(os.getenv("HF_API_KEY",         ""))
    OPENROUTER_API_KEY: str = _clean(os.getenv("OPENROUTER_API_KEY", ""))

    DEFAULT_HF_LLM: str = "meta-llama/Llama-3.1-8B-Instruct"
    DEFAULT_HF_VLM: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    DEFAULT_OR_LLM: str = "mistralai/mistral-7b-instruct:free"
    DEFAULT_OR_VLM: str = "qwen/qwen-2.5-vl-7b-instruct:free"

    #Auto-detect which backend to use
    # Priority: HF_TOKEN > OPENROUTER_API_KEY
    @classmethod
    def get_api_key(cls) -> str:
        """Return whichever API key is configured."""
        if cls.HF_TOKEN and cls.HF_TOKEN not in ("hf_your_token_here", ""):
            return cls.HF_TOKEN
        if cls.HF_API_KEY and cls.HF_API_KEY not in ("hf_your_token_here", ""):
            return cls.HF_API_KEY
        return cls.OPENROUTER_API_KEY

    @classmethod
    def get_base_url(cls) -> str:
        """
        Return the correct API base URL based on the token type.
          hf_...    → HuggingFace Inference Providers router
          sk-or-... → OpenRouter
          anything else → HuggingFace router (safest default)
        """
        key = cls.get_api_key()
        if key.startswith("sk-or-"):
            return "https://openrouter.ai/api/v1"
        # HuggingFace token (hf_...) or anything else
        return "https://router.huggingface.co/v1"

    @classmethod
    def get_backend_name(cls) -> str:
        key = cls.get_api_key()
        return "OpenRouter" if key.startswith("sk-or-") else "HuggingFace"

    # Model Names
    # Optional overrides. If left blank, backend-safe defaults are chosen
    # automatically based on the configured API key.
    LLM_MODEL: str = _clean(os.getenv("LLM_MODEL", ""))
    VLM_MODEL: str = _clean(os.getenv("VLM_MODEL", ""))

    # Run locally — no API needed
    EMBEDDING_MODEL: str = _clean(os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"))
    RERANKER_MODEL:  str = _clean(os.getenv("RERANKER_MODEL",  "cross-encoder/ms-marco-MiniLM-L-6-v2"))

    #  Storage (Windows-safe paths)
    _base          = Path(__file__).resolve().parent
    CHROMA_DB_PATH: str = str(_base / "data" / "chroma_db")
    UPLOAD_DIR:     str = str(_base / "data" / "uploads")

    #  Chunking 
    CHUNK_SIZE:    int = int(_clean(os.getenv("CHUNK_SIZE",    "400")))
    CHUNK_OVERLAP: int = int(_clean(os.getenv("CHUNK_OVERLAP", "60")))

    #  Retrieval
    BM25_TOP_K:     int = int(_clean(os.getenv("BM25_TOP_K",     "15")))
    DENSE_TOP_K:    int = int(_clean(os.getenv("DENSE_TOP_K",    "15")))
    RERANKER_TOP_K: int = int(_clean(os.getenv("RERANKER_TOP_K", "5")))

    @classmethod
    def get_llm_model(cls) -> str:
        """Return the configured LLM override or a backend-safe default."""
        if cls.LLM_MODEL:
            return cls.LLM_MODEL
        if cls.get_backend_name() == "OpenRouter":
            return cls.DEFAULT_OR_LLM
        return cls.DEFAULT_HF_LLM

    @classmethod
    def get_vlm_model(cls) -> str:
        """Return the configured VLM override or a backend-safe default."""
        if cls.VLM_MODEL:
            return cls.VLM_MODEL
        if cls.get_backend_name() == "OpenRouter":
            return cls.DEFAULT_OR_VLM
        return cls.DEFAULT_HF_VLM

    @classmethod
    def get_llm_fallback_models(cls) -> list[str]:
        """Return a short ordered list of alternative LLMs for retries."""
        if cls.get_backend_name() == "OpenRouter":
            return [
                cls.DEFAULT_OR_LLM,
                "meta-llama/llama-3.3-70b-instruct:free",
                "google/gemma-3-27b-it:free",
            ]
        return [
            cls.DEFAULT_HF_LLM,
            "google/gemma-2-9b-it",
            "mistralai/Mistral-7B-Instruct-v0.3",
        ]

    @classmethod
    def get_vlm_fallback_models(cls) -> list[str]:
        """Return a short ordered list of alternative VLMs for retries."""
        if cls.get_backend_name() == "OpenRouter":
            return [
                cls.DEFAULT_OR_VLM,
                "meta-llama/llama-3.2-11b-vision-instruct:free",
            ]
        return [
            cls.DEFAULT_HF_VLM,
            "meta-llama/Llama-3.2-11B-Vision-Instruct",
        ]

    # App 
    SUPPORTED_FORMATS: list = [
        "pdf", "docx", "doc", "txt",
        "xlsx", "xls", "csv",
        "png", "jpg", "jpeg", "pptx"
    ]

    @classmethod
    def validate(cls) -> bool:
        """Check that at least one API key is configured."""
        key = cls.get_api_key()
        if not key or key in ("hf_your_token_here", "sk-or-your-key-here", ""):
            print("[Config] ❌ No API key found.")
            print("  Add HF_API_KEY=hf_... or HF_TOKEN=hf_... to your .env file")
            print("  Get a free token at: https://huggingface.co/settings/tokens")
            return False
        return True

    @classmethod
    def debug(cls):
        """Print current config — masks the API key."""
        key    = cls.get_api_key()
        masked = f"{key[:10]}...{key[-4:]}" if len(key) > 14 else ("SET" if key else "NOT SET")
        print(f"  Backend      : {cls.get_backend_name()}")
        print(f"  Base URL     : {cls.get_base_url()}")
        print(f"  API Key      : {masked}")
        print(f"  LLM_MODEL    : {cls.get_llm_model()}")
        print(f"  VLM_MODEL    : {cls.get_vlm_model()}")
        print(f"  EMBEDDING    : {cls.EMBEDDING_MODEL}")
        print(f"  CHROMA_PATH  : {cls.CHROMA_DB_PATH}")
        print(f"  CHUNK_SIZE   : {cls.CHUNK_SIZE}")
