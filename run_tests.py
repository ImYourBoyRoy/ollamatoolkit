# ./run_tests.py
"""
OllamaToolkit Test Runner
=========================
Runs all tests with coverage and detailed output.

Usage:
    python run_tests.py           # Run all tests
    python run_tests.py -v        # Verbose mode
    python run_tests.py --cov     # With coverage
"""

import subprocess
import sys
import shutil
from pathlib import Path


def clean_pycache():
    """Remove __pycache__ directories for test freshness."""
    root = Path(__file__).parent
    for cache_dir in root.rglob("__pycache__"):
        shutil.rmtree(cache_dir, ignore_errors=True)
    print("[OK] Cleaned __pycache__")


def run_tests():
    """Run pytest with appropriate flags."""
    clean_pycache()

    # Build command
    cmd = [sys.executable, "-m", "pytest", "tests/", "-v", "--tb=short"]

    # Add coverage if requested
    if "--cov" in sys.argv:
        cmd.extend(["--cov=src/ollamatoolkit", "--cov-report=term-missing"])

    # Pass through other args
    for arg in sys.argv[1:]:
        if arg not in ["--cov"]:
            cmd.append(arg)

    print(f"\n[TEST] Running: {' '.join(cmd)}\n")
    result = subprocess.run(cmd, cwd=Path(__file__).parent)
    return result.returncode


if __name__ == "__main__":
    sys.exit(run_tests())
