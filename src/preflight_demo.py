from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

from src.config import settings


def _check_python() -> tuple[bool, str]:
    major, minor = sys.version_info.major, sys.version_info.minor
    ok = major == 3 and minor >= 10
    return ok, f"Python {major}.{minor}"


def _check_module(module_name: str, label: str) -> tuple[bool, str]:
    try:
        __import__(module_name)
        return True, f"{label} installed"
    except Exception as exc:
        return False, f"{label} missing ({exc})"


def _check_ollama(host: str, model: str) -> tuple[bool, str]:
    url = host.rstrip("/") + "/api/tags"
    req = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=8) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        return False, f"Ollama unreachable at {host} ({exc})"
    except Exception as exc:
        return False, f"Ollama health check failed ({exc})"

    names = [item.get("name", "") for item in payload.get("models", []) if isinstance(item, dict)]
    has_model = any(name == model or name.startswith(model + ":") for name in names)
    if has_model:
        return True, f"Ollama up and model {model} available"
    return False, f"Ollama up but model {model} not found"


def _check_model_artifact(model_path: Path) -> tuple[bool, str]:
    if model_path.exists():
        return True, f"Model artifact found at {model_path}"
    return False, f"Model artifact missing at {model_path}"


def run_preflight(ollama_host: str, ollama_model: str) -> int:
    checks = [
        ("python",) + _check_python(),
        ("crewai",) + _check_module("crewai", "CrewAI"),
        ("streamlit",) + _check_module("streamlit", "Streamlit"),
        ("trained_model",) + _check_model_artifact(settings.model_path_abs),
        ("ollama_backend",) + _check_ollama(ollama_host, ollama_model),
    ]

    all_ok = all(ok for _, ok, _ in checks)
    banner = "PRECHECK PASS" if all_ok else "PRECHECK FAIL"
    print("=" * 64)
    print(banner)
    print("=" * 64)
    for name, ok, detail in checks:
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {name}: {detail}")

    return 0 if all_ok else 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Demo readiness preflight for CrewAI + Ollama runtime.")
    parser.add_argument("--ollama-host", default=settings.ollama_host, help="Ollama base URL.")
    parser.add_argument("--ollama-model", default=settings.ollama_model, help="Ollama model name.")
    args = parser.parse_args()
    raise SystemExit(run_preflight(args.ollama_host, args.ollama_model))


if __name__ == "__main__":
    main()
