
# run.py — Project Launcher
#
# Usage:
#   python run.py              → start the Streamlit app
#   python run.py --setup      → check installation
#   python run.py --test       → run unit tests
#   python run.py --config     → print current config


import sys
import os
import subprocess
import argparse

# Suppress noisy warnings before any imports 
# These environment variables silence TensorFlow and other
# libraries that print warnings even when not being used.
os.environ["TF_CPP_MIN_LOG_LEVEL"]    = "3"   # suppress TF C++ logs
os.environ["TF_ENABLE_ONEDNN_OPTS"]   = "0"   # suppress oneDNN messages
os.environ["TOKENIZERS_PARALLELISM"]  = "false"  # suppress tokenizer warnings
os.environ["TRANSFORMERS_VERBOSITY"]  = "error"  # suppress transformers info logs

import warnings
warnings.filterwarnings("ignore")  # suppress all Python warnings during setup


def check_setup():
    """
    Pre-flight checks — run this once after installation.
    Checks: .env file, API key, all required packages, spaCy model.
    """
    print("\n" + "=" * 55)
    print("  AI Document Intelligence — Setup Check")
    print("=" * 55 + "\n")

    all_ok = True

    # 1. Check .env file
    from pathlib import Path
    env_path = Path(__file__).parent / ".env"

    if env_path.exists():
        print("  ✅ .env file found")
    else:
        print("  ❌ .env file NOT found")
        print("     Fix: In your project folder, run:")
        print("          copy .env.example .env       (Windows CMD)")
        print("          cp .env.example .env         (Mac/Linux)")
        print("     Then open .env and add your API key.")
        all_ok = False

    #  2. Check API key 
    from config import Config

    # key = Config.API_KEY
    key = Config.get_api_key()
    if key and key not in ("sk-or-your-key-here", "your-api-key-here", ""):
        masked   = f"{key[:10]}...{key[-4:]}" if len(key) > 14 else key
        # provider = Config.PROVIDER
        provider = Config.get_backend_name()
        print(f"  ✅ API key configured — provider: {provider} ({masked})")
        print(f"     LLM : {Config.get_llm_model()}")
        print(f"     VLM : {Config.get_vlm_model()}")
        print(f"     URL : {Config.get_base_url()}")
    else:
        print("  ❌ API key NOT set")
        print()
        print("     Option A — HuggingFace (your current setup):")
        print("       1. Go to https://huggingface.co/settings/tokens")
        print("       2. Create a token (read access is enough)")
        print("       3. Add to .env (NO quotes): HF_API_KEY=hf_...")
        print()
        print("     Option B — OpenRouter:")
        print("       1. Go to https://openrouter.ai")
        print("       2. Create a free API key")
        print("       3. Add to .env (NO quotes): OPENROUTER_API_KEY=sk-or-...")
        print()
        print('     Common mistake (WRONG):  OPENROUTER_API_KEY="hf_..."')
        print("     Correct format  (RIGHT): HF_API_KEY=hf_...")
        all_ok = False

    # 3. Check core packages
    packages = [
        ("pymupdf",                 "fitz",                "PDF parsing"),
        ("python-docx",             "docx",                "Word documents"),
        ("openpyxl",                "openpyxl",            "Excel files"),
        ("openai",                  "openai",              "OpenRouter API client"),
        ("sentence-transformers",   "sentence_transformers","Embeddings + reranker"),
        ("chromadb",                "chromadb",            "Vector database"),
        ("rank-bm25",               "rank_bm25",           "BM25 retrieval"),
        ("langchain-text-splitters","langchain_text_splitters","LangChain chunking"),
        ("streamlit",               "streamlit",           "Web UI"),
        ("python-dotenv",           "dotenv",              ".env file reader"),
        ("loguru",                  "loguru",              "Logging"),
        ("Pillow",                  "PIL",                 "Image processing"),
    ]

    print()
    print("  Checking packages...")
    for pkg_name, import_name, description in packages:
        try:
            __import__(import_name)
            print(f"  ✅ {pkg_name:<30} ({description})")
        except ImportError:
            print(f"  ❌ {pkg_name:<30} MISSING — run: pip install {pkg_name}")
            all_ok = False

    # 4. Check optional packages
    optional = [
        ("spacy",  "spacy",  "NER + sentence splitting (recommended)"),
    ]
    print()
    print("  Checking optional packages...")
    for pkg_name, import_name, description in optional:
        try:
            __import__(import_name)
            print(f"  ✅ {pkg_name:<30} ({description})")
        except ImportError:
            print(f"  ⚠️  {pkg_name:<30} not installed (optional but recommended)")
            print(f"       Install: pip install {pkg_name}")
            print(f"       Then:    python -m spacy download en_core_web_sm")

    #  5. Check spaCy model 
    print()
    try:
        import spacy
        spacy.load("en_core_web_sm")
        print("  ✅ spaCy en_core_web_sm model found")
    except ImportError:
        print("  ⚠️  spaCy not installed — using basic sentence splitting (OK)")
    except OSError:
        print("  ⚠️  spaCy model not downloaded yet")
        print("       Fix: python -m spacy download en_core_web_sm")

    # Final result 
    print()
    print("=" * 55)
    if all_ok:
        print("  🎉 All checks passed!")
        print()
        print("  Start the app with:")
        print("      python run.py")
        print()
        print("  Then open: http://localhost:8501")
    else:
        print("  ⚠️  Some issues need fixing (see above).")
        print("  After fixing, run this check again:")
        print("      python run.py --setup")
    print("=" * 55 + "\n")


def run_app():
    """Launch the Streamlit app."""
    print("\n🚀 Starting AI Document Intelligence System...")
    print("   Open in browser: http://localhost:8501")
    print("   Press Ctrl+C to stop.\n")

    subprocess.run([
        sys.executable, "-m", "streamlit", "run",
        os.path.join("ui", "app.py"),
        "--server.headless=false",
        "--browser.gatherUsageStats=false",
        "--server.port=8501",
    ])


def run_tests():
    """Run unit tests."""
    print("\n🧪 Running unit tests...\n")
    result = subprocess.run([
        sys.executable, "-m", "pytest", "tests/", "-v", "--tb=short"
    ])
    sys.exit(result.returncode)


def print_config():
    """Print current configuration (useful for debugging)."""
    print("\n⚙️  Current Configuration\n")
    from config import Config
    Config.debug()
    print()


# CLI Entry Point
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="AI Document Intelligence System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Commands:
  python run.py              Start the web app (default)
  python run.py --setup      Check installation & API key
  python run.py --config     Print current configuration
  python run.py --test       Run unit tests
        """
    )
    parser.add_argument("--setup",  action="store_true", help="Check installation")
    parser.add_argument("--config", action="store_true", help="Print config")
    parser.add_argument("--test",   action="store_true", help="Run unit tests")

    args = parser.parse_args()

    if args.setup:
        check_setup()
    elif args.config:
        print_config()
    elif args.test:
        run_tests()
    else:
        run_app()
