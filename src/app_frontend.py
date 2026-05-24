from __future__ import annotations

import csv
import io
import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from datetime import datetime, timezone
from typing import Any
import sys

import streamlit as st

from src.agents.orchestrator import run_screening
from src.agents.schemas import ScreeningInput
from src.crew_runtime import is_crewai_available
from src.config import settings
from src.llm_client import OllamaClient
from src.pipeline.batch import run_batch_screening
from src.tools.skill_extractor_tool import extract_skills
from src.utils.json_logger import JsonLogger


# ── PDF helper (optional dep) ────────────────────────────────────────────────

def _extract_pdf_text(file_bytes: bytes) -> str:
    try:
        from pdfminer.high_level import extract_text_to_fp
        from pdfminer.layout import LAParams
        out = io.StringIO()
        extract_text_to_fp(io.BytesIO(file_bytes), out, laparams=LAParams(), output_type="text", codec="utf-8")
        return out.getvalue().strip()
    except Exception as exc:
        return f"[PDF extraction error: {exc}]"


# ── Explainability helper ────────────────────────────────────────────────────

_SIGNAL_WORDS = [
    "python", "pytorch", "tensorflow", "sql", "spark", "docker", "kubernetes",
    "machine learning", "deep learning", "nlp", "cloud", "azure", "aws", "gcp",
    "senior", "lead", "manager", "director", "principal",
    "bachelor", "master", "phd", "engineer",
    "project", "deployed", "built", "delivered", "launched",
    "team", "collaboration", "stakeholder", "communication",
    "pandas", "scikit", "fastapi", "transformers", "huggingface", "airflow",
]

def _highlight_cv(cv_text: str, job_text: str) -> str:
    """Return HTML with matched signal words highlighted."""
    from src.data.dataset import extract_skills as _extract
    job_skills = _extract(job_text)
    signals = {w.lower() for w in _SIGNAL_WORDS} | {s.lower() for s in job_skills}
    words = cv_text.split()
    out: list[str] = []
    for word in words:
        clean = word.lower().strip(".,;:()[]\"'")
        if clean in signals:
            out.append(
                f"<mark style='background:#d4f5e9;border-radius:3px;padding:1px 3px;"
                f"font-weight:600;color:#093028'>{word}</mark>"
            )
        else:
            out.append(word)
    return " ".join(out)


# ── Styles ───────────────────────────────────────────────────────────────────

def _inject_styles() -> None:
    st.markdown(
        """
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;600;700;800&family=IBM+Plex+Mono:wght@400;500&display=swap');
            :root {
                --bg-start: #f4f8f2;
                --bg-end: #e5f2ff;
                --ink: #11202b;
                --muted: #4a5b67;
                --accent: #007a70;
                --accent-2: #f4a259;
                --card: #ffffff;
                --line: #d9e6ee;
                --soft: #f8fbfd;
            }
            html, body, [class*="css"] {
                font-family: 'Manrope', sans-serif;
                color: var(--ink);
            }
            .block-container {
                padding-top: 1.25rem;
                padding-bottom: 2rem;
                max-width: 1280px;
            }
            .stApp {
                background:
                    radial-gradient(circle at top left, rgba(255,255,255,0.86), rgba(255,255,255,0) 28%),
                    radial-gradient(circle at bottom right, rgba(195, 226, 255, 0.34), rgba(255,255,255,0) 26%),
                    linear-gradient(180deg, var(--bg-start) 0%, var(--bg-end) 100%);
            }
            [data-testid="stHeader"] {
                background: transparent;
                height: 0;
            }
            [data-testid="stToolbar"],
            [data-testid="stDecoration"],
            [data-testid="stStatusWidget"] {
                display: none;
            }
            .stApp,
            .stMarkdown,
            .stMarkdown p,
            .stMarkdown li,
            .stMarkdown label,
            .stRadio label,
            .stRadio label span,
            .stRadio [role="radiogroup"] label,
            .stRadio [role="radiogroup"] label span,
            .stCheckbox label,
            .stCheckbox label span,
            .stToggle label,
            .stToggle label span,
            .stTextInput label,
            .stTextArea label,
            .stSelectbox label,
            .stFileUploader label,
            [data-testid="stWidgetLabel"],
            [data-testid="stWidgetLabel"] *,
            [data-baseweb="radio"],
            [data-baseweb="radio"] *,
            [data-testid="stSidebar"],
            [data-testid="stSidebar"] p,
            [data-testid="stSidebar"] li,
            [data-testid="stSidebar"] label,
            [data-testid="stSidebar"] span {
                color: var(--ink) !important;
            }
            .stApp h1,
            .stApp h2,
            .stApp h3,
            .stApp h4,
            .stApp h5,
            .stApp h6,
            [data-testid="stSidebar"] h1,
            [data-testid="stSidebar"] h2,
            [data-testid="stSidebar"] h3 {
                color: var(--ink) !important;
            }
            .hero,
            .hero *,
            .hero h1,
            .hero p,
            .hero span {
                color: #f8fffb !important;
            }
            [data-testid="stSidebar"] small,
            [data-testid="stSidebar"] [data-testid="stCaptionContainer"],
            [data-testid="stSidebar"] [data-testid="stCaptionContainer"] * {
                color: var(--muted) !important;
            }
            h1, h2, h3 {
                letter-spacing: -0.03em;
            }
            .hero {
                background:
                    radial-gradient(circle at top right, rgba(255,255,255,0.12), rgba(255,255,255,0) 22%),
                    linear-gradient(135deg, #092f2a 0%, #13564e 52%, #237a57 100%);
                padding: 1.35rem 1.35rem 1.1rem;
                border-radius: 20px;
                color: #f8fffb;
                margin-bottom: 1.15rem;
                box-shadow: 0 18px 38px rgba(6, 40, 34, 0.18);
                border: 1px solid rgba(255,255,255,0.1);
            }
            .hero-kicker {
                display: inline-flex;
                align-items: center;
                gap: 0.35rem;
                padding: 0.28rem 0.62rem;
                border-radius: 999px;
                background: rgba(255,255,255,0.12);
                border: 1px solid rgba(255,255,255,0.16);
                font-size: 0.76rem;
                font-weight: 800;
                text-transform: uppercase;
                letter-spacing: 0.08em;
            }
            .hero h1 {
                margin: 0.7rem 0 0;
                font-size: 2rem;
                line-height: 1.04;
                font-weight: 800;
                max-width: 12ch;
            }
            .hero p  {
                margin: 0.55rem 0 0;
                opacity: 0.94;
                max-width: 60ch;
                font-size: 1rem;
            }
            .hero-pills {
                display: flex;
                flex-wrap: wrap;
                gap: 0.55rem;
                margin-top: 0.95rem;
            }
            .hero-pill {
                display: inline-flex;
                align-items: center;
                padding: 0.42rem 0.7rem;
                border-radius: 999px;
                background: rgba(255,255,255,0.14);
                border: 1px solid rgba(255,255,255,0.12);
                font-size: 0.84rem;
                font-weight: 700;
            }
            .section-intro {
                margin: 0.5rem 0 0.85rem;
                padding: 0.85rem 1rem;
                background: rgba(255,255,255,0.72);
                border: 1px solid rgba(17, 32, 43, 0.08);
                border-radius: 16px;
                box-shadow: 0 10px 24px rgba(17, 32, 43, 0.05);
                backdrop-filter: blur(6px);
            }
            .section-eyebrow {
                font-size: 0.76rem;
                font-weight: 800;
                text-transform: uppercase;
                letter-spacing: 0.08em;
                color: var(--accent);
                margin-bottom: 0.22rem;
            }
            .section-title {
                margin: 0;
                font-size: 1.2rem;
                font-weight: 800;
                color: var(--ink) !important;
            }
            .section-copy {
                margin: 0.32rem 0 0;
                color: var(--muted) !important;
                font-size: 0.94rem;
            }
            .metric {
                background: var(--card);
                border: 1px solid var(--line);
                border-radius: 16px;
                padding: 0.9rem 0.95rem;
                box-shadow: 0 10px 22px rgba(17, 32, 43, 0.06);
                color: var(--ink);
            }
            .metric-label {
                color: var(--muted);
                font-size: 0.74rem;
                font-weight: 800;
                text-transform: uppercase;
                letter-spacing: 0.08rem;
            }
            .metric-value {
                font-size: 1.45rem;
                font-weight: 800;
                color: var(--ink);
                margin-top: 0.2rem;
                line-height: 1.05;
            }
            .mono { font-family: 'IBM Plex Mono', monospace; }
            .skill-match { color: #007a70; font-weight: 700; }
            .skill-miss  { color: #c0392b; font-weight: 700; }
            .history-item {
                border-left: 3px solid #237a57;
                padding: 0.55rem 0.8rem;
                margin-bottom: 0.48rem;
                background: #eef8f3;
                border-radius: 0 10px 10px 0;
                font-size: 0.85rem;
                color: var(--ink);
            }
            .showcase-card {
                background: linear-gradient(145deg, #ffffff, #e7f3ef);
                border: 1px solid #d3e8e2;
                border-radius: 18px;
                padding: 1rem;
                box-shadow: 0 12px 26px rgba(10, 63, 49, 0.08);
                animation: fadeInUp 0.45s ease;
                color: var(--ink);
                min-height: 132px;
            }
            .showcase-label {
                color: #45606a;
                font-size: 0.74rem;
                font-weight: 800;
                text-transform: uppercase;
                letter-spacing: 0.08rem;
            }
            .showcase-value {
                font-size: 1.7rem;
                font-weight: 800;
                color: #0a3f31;
                margin-top: 0.28rem;
                line-height: 1.02;
            }
            .showcase-sub {
                font-size: 0.86rem;
                color: #4a5b67;
                margin-top: 0.3rem;
            }
            .brief-ok {
                background: #e6f6ef;
                border-left: 4px solid #007a70;
                padding: 0.7rem 0.8rem;
                border-radius: 10px;
                margin-bottom: 0.55rem;
                font-size: 0.92rem;
                color: var(--ink);
                box-shadow: inset 0 0 0 1px rgba(0, 122, 112, 0.08);
            }
            .stTabs [data-baseweb="tab-list"] {
                gap: 0.45rem;
                background: rgba(255,255,255,0.5);
                padding: 0.35rem;
                border-radius: 14px;
                border: 1px solid rgba(17, 32, 43, 0.08);
            }
            .stTabs [data-baseweb="tab"] {
                height: auto;
                padding: 0.55rem 0.9rem;
                border-radius: 10px;
                font-weight: 700;
                color: var(--muted);
            }
            .stTabs [aria-selected="true"] {
                background: #ffffff;
                color: var(--ink);
                box-shadow: 0 6px 16px rgba(17, 32, 43, 0.08);
            }
            .stButton > button, .stDownloadButton > button {
                border-radius: 12px;
                font-weight: 700;
                border: 1px solid rgba(17, 32, 43, 0.08);
                box-shadow: 0 8px 18px rgba(17, 32, 43, 0.06);
            }
            [data-testid="stSidebar"] {
                border-right: 1px solid rgba(17, 32, 43, 0.06);
                background: rgba(255,255,255,0.62);
                backdrop-filter: blur(6px);
            }
            .brief-ok b,
            .brief-ok strong,
            .brief-ok span,
            .brief-ok div,
            .brief-ok p,
            .showcase-card b,
            .showcase-card strong,
            .history-item b,
            .history-item strong,
            .metric b,
            .metric strong {
                color: inherit;
            }
            @keyframes fadeInUp {
                from { opacity: 0; transform: translateY(8px); }
                to { opacity: 1; transform: translateY(0); }
            }
            @media (max-width: 900px) {
                .hero h1 {
                    font-size: 1.55rem;
                    max-width: none;
                }
                .section-intro {
                    padding: 0.8rem 0.9rem;
                }
                .showcase-card {
                    min-height: auto;
                }
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_section_intro(title: str, copy: str, eyebrow: str = "Workspace") -> None:
    st.markdown(
        f"""
        <div class="section-intro">
            <div class="section-eyebrow">{eyebrow}</div>
            <h2 class="section-title">{title}</h2>
            <p class="section-copy">{copy}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ── Data helpers ─────────────────────────────────────────────────────────────

def _load_sample_data(root: Path) -> tuple[dict, dict]:
    cv = json.loads((root / "data" / "sample_cvs.json").read_text(encoding="utf-8"))[0]
    job = json.loads((root / "data" / "sample_jobs.json").read_text(encoding="utf-8"))[0]
    return cv, job


def _load_all_jobs(root: Path) -> list[dict]:
    return json.loads((root / "data" / "sample_jobs.json").read_text(encoding="utf-8"))


def _load_all_cvs(root: Path) -> list[dict]:
    return json.loads((root / "data" / "sample_cvs.json").read_text(encoding="utf-8"))


def _load_json_from_upload(uploaded_file) -> list[dict] | dict | None:
    if uploaded_file is None:
        return None
    try:
        payload = json.loads(uploaded_file.getvalue().decode("utf-8"))
    except Exception:
        return None
    return payload


def _candidate_id_from_filename(filename: str, index: int) -> str:
    stem = Path(filename).stem.strip().lower()
    cleaned = "".join(ch if ch.isalnum() else "_" for ch in stem).strip("_")
    return cleaned or f"uploaded_pdf_{index:03d}"


def _make_unique_candidate_id(candidate_id: str, seen: set[str], index: int) -> str:
    base = candidate_id.strip() or f"candidate_{index:03d}"
    unique = base
    suffix = 2
    while unique in seen:
        unique = f"{base}_{suffix}"
        suffix += 1
    seen.add(unique)
    return unique


def _ensure_unique_candidate_ids(cvs: list[dict]) -> list[dict]:
    seen: set[str] = set()
    normalized: list[dict] = []
    for index, cv in enumerate(cvs, start=1):
        record = dict(cv)
        raw_id = _as_text(record.get("candidate_id", f"candidate_{index:03d}"), f"candidate_{index:03d}")
        record["candidate_id"] = _make_unique_candidate_id(raw_id, seen, index)
        normalized.append(record)
    return normalized


def _load_cvs_from_pdf_uploads(uploaded_files: list[Any]) -> list[dict[str, str]]:
    cvs: list[dict[str, str]] = []
    for index, uploaded_file in enumerate(uploaded_files, start=1):
        text = _extract_pdf_text(uploaded_file.getvalue())
        extraction_ok = bool(text.strip()) and not text.startswith("[PDF extraction error:")
        cvs.append(
            {
                "candidate_id": _candidate_id_from_filename(uploaded_file.name, index),
                "cv_text": text,
                "source_file": uploaded_file.name,
                "extraction_status": "success" if extraction_ok else "failure",
            }
        )
    return _ensure_unique_candidate_ids(cvs)



def _read_json_file(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _as_text(value: Any, default: str = "") -> str:
    if value is None:
        return default
    return str(value)


def _collect_defense_evidence(root: Path) -> dict[str, Any]:
    model_eval = _read_json_file(root / "models" / "model_evaluation.json") or {}
    pipeline_eval = _read_json_file(root / "logs" / "pipeline_evaluation.json") or {}
    n_jobs = len(_load_all_jobs(root))
    n_cvs = len(_load_all_cvs(root))
    logs_count = len(list((root / "logs").glob("run_*.jsonl")))

    return {
        "model_accuracy": float(model_eval.get("accuracy", 0.0)),
        "model_macro_f1": float(model_eval.get("macro_f1", 0.0)),
        "pipeline_accuracy": float((pipeline_eval.get("metrics") or {}).get("accuracy", 0.0)),
        "pipeline_cases": int(pipeline_eval.get("n_cases", 0)),
        "n_jobs": n_jobs,
        "n_cvs": n_cvs,
        "logs_count": logs_count,
        "has_model": (root / "models" / "cv_fit_model.pt").exists(),
        "has_pipeline_eval": (root / "logs" / "pipeline_evaluation.json").exists(),
        "has_tests": (root / "tests").exists(),
    }


def _render_showcase_cards(evidence: dict[str, Any]) -> None:
    c1, c2, c3, c4 = st.columns(4)
    cards = [
        ("Model Accuracy", f"{evidence['model_accuracy']:.1%}", "PyTorch classifier"),
        ("Pipeline Accuracy", f"{evidence['pipeline_accuracy']:.1%}", f"{evidence['pipeline_cases']} labeled cases"),
        ("Dataset Coverage", f"{evidence['n_cvs']} CV / {evidence['n_jobs']} Jobs", "Single + batch demo ready"),
        ("Traceability", f"{evidence['logs_count']} run logs", "JSONL runtime evidence"),
    ]
    for col, (label, value, sub) in zip([c1, c2, c3, c4], cards):
        col.markdown(
            f"<div class='showcase-card'><div class='showcase-label'>{label}</div>"
            f"<div class='showcase-value'>{value}</div><div class='showcase-sub'>{sub}</div></div>",
            unsafe_allow_html=True,
        )


def _render_brief_alignment(evidence: dict[str, Any]) -> None:
    with st.expander("Defense Checklist - brief alignment evidence", expanded=True):
        checks = [
            ("Multi-agent system", "Technical Matcher + Profile Analyzer + Orchestrator", True),
            ("Deep learning model integration", "PyTorch model used in scoring pipeline", evidence["has_model"]),
            ("Tools integration", "Skill extractor + model tool exposed in agent flow", True),
            ("Human-in-the-loop", "Borderline and conflict review checkpoint", True),
            ("Evaluation & robustness", "Unit tests + pipeline evaluation artifacts", evidence["has_tests"] and evidence["has_pipeline_eval"]),
            ("Traceability", "Per-run JSON logs and downloadable reports", evidence["logs_count"] > 0),
            ("Presentation readiness", "Single CV, batch CV, PDF upload, explainability", True),
        ]
        for title, detail, ok in checks:
            status = "PASS" if ok else "MISSING"
            tone = "#007a70" if ok else "#c0392b"
            st.markdown(
                f"<div class='brief-ok'><b>{title}</b> - <span style='color:{tone};font-weight:700'>{status}</span><br>{detail}</div>",
                unsafe_allow_html=True,
            )


def _render_runtime_status(root: Path) -> None:
    st.sidebar.divider()
    st.sidebar.subheader("Agent Runtime")

    py_version = f"{sys.version_info.major}.{sys.version_info.minor}"
    venv312_exists = (root / ".venv312" / "Scripts" / "python.exe").exists()
    crewai_ready = is_crewai_available()

    if crewai_ready:
        st.sidebar.success("CrewAI primary: ACTIVE")
    else:
        st.sidebar.warning("CrewAI primary: NOT ACTIVE in current Python")

    if venv312_exists:
        st.sidebar.info("CrewAI-ready env detected: .venv312 (Python 3.12)")

    st.sidebar.caption(f"Current app runtime: deterministic frontend pipeline (Python {py_version})")
    st.sidebar.caption("CLI supports CrewAI-first mode via: python -m src.main --runtime auto --ollama")


def _render_quick_mode_guide() -> None:
    with st.sidebar.expander("Quick Guide: modes and input types", expanded=False):
        st.markdown("**Input types**")
        st.markdown("- Sample = demo data")
        st.markdown("- Manual = typed data")
        st.markdown("- PDF = document parsing")
        st.markdown("- JSON = structured machine data")

        st.markdown("**Screening modes**")
        st.markdown("- Single = 1 CV to 1 job")
        st.markdown("- Batch = many CVs to 1 job")


def _resolve_ollama_executable() -> str | None:
    cmd = shutil.which("ollama")
    if cmd:
        return cmd
    local = os.getenv("LOCALAPPDATA")
    if not local:
        return None
    candidate = Path(local) / "Programs" / "Ollama" / "ollama.exe"
    if candidate.exists():
        return str(candidate)
    return None


def _start_ollama_server() -> tuple[bool, str]:
    exe = _resolve_ollama_executable()
    if not exe:
        return False, "Ollama executable not found. Install it first."
    try:
        flags = 0
        if os.name == "nt":
            flags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS
        subprocess.Popen([exe, "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, creationflags=flags)
        time.sleep(2)
        probe = OllamaClient(host=settings.ollama_host, model=settings.ollama_model)
        if probe.is_available():
            return True, f"Ollama is running at {settings.ollama_host}"
        return False, "Ollama process started but endpoint is not responding yet."
    except Exception as exc:
        return False, f"Failed to start Ollama: {exc}"


def _pull_ollama_model(model_name: str) -> tuple[bool, str]:
    exe = _resolve_ollama_executable()
    if not exe:
        return False, "Ollama executable not found."
    try:
        result = subprocess.run([exe, "pull", model_name], capture_output=True, text=True, timeout=1800)
        if result.returncode == 0:
            return True, f"Model '{model_name}' is ready."
        msg = (result.stderr or result.stdout).strip() or "Unknown pull error"
        return False, msg
    except Exception as exc:
        return False, f"Failed pulling model: {exc}"


# ── CSV builder ──────────────────────────────────────────────────────────────

def _results_to_csv(results: list[dict]) -> str:
    buf = io.StringIO()
    cols = ["rank", "candidate_id", "job_id", "final_label", "final_score",
            "recommendation", "tech_score", "profile_score"]
    writer = csv.DictWriter(buf, fieldnames=cols, extrasaction="ignore")
    writer.writeheader()
    for i, r in enumerate(results, start=1):
        writer.writerow({
            "rank": i,
            "candidate_id": r.get("candidate_id", ""),
            "job_id": r.get("job_id", ""),
            "final_label": r.get("final_label", ""),
            "final_score": r.get("final_score", ""),
            "recommendation": r.get("recommendation", ""),
            "tech_score": r.get("technical", {}).get("score", ""),
            "profile_score": r.get("profile", {}).get("score", ""),
        })
    return buf.getvalue()


# ── Single screening run ─────────────────────────────────────────────────────

def _run_one(
    cv_text: str,
    job_text: str,
    candidate_id: str,
    job_id: str,
    logger: JsonLogger,
    ollama_client: OllamaClient | None,
    require_human_approval: bool = False,
) -> dict[str, Any]:
    payload = ScreeningInput(
        candidate_id=candidate_id.strip(),
        cv_text=cv_text.strip(),
        job_id=job_id.strip(),
        job_text=job_text.strip(),
    )
    return run_screening(
        screening_input=payload,
        model_path=settings.model_path_abs,
        low=settings.borderline_low,
        high=settings.borderline_high,
        logger=logger,
        ollama_client=ollama_client,
        interactive_human_review=False,
        require_human_approval=require_human_approval,
    )


def _apply_manual_human_decision(
    result: dict[str, Any],
    logger: JsonLogger,
    decision: str,
    reviewer: str,
    comment: str = "",
) -> dict[str, Any]:
    status_map = {
        "Approve": "approved",
        "Shortlist": "approved",
        "Reject": "rejected",
        "Flag": "flagged",
        "Needs Review": "flagged",
    }
    recommendation_map = {
        "approved": "shortlist",
        "rejected": "reject",
        "flagged": "review",
    }

    status = status_map[decision]
    manual_review = {
        "candidate_id": result.get("candidate_id"),
        "score": result.get("final_score"),
        "status": status,
        "reviewer": reviewer.strip() or "anonymous",
        "comment": comment.strip(),
        "reason": "manual approval captured in Streamlit",
        "reasons": (result.get("human_review") or {}).get("reasons", result.get("review_reasons", [])),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    result["human_review"] = manual_review
    result["recommendation"] = recommendation_map[status]

    logger.log(
        "human_checkpoint",
        manual_review,
        agent_name="human_reviewer",
        action="manual_review_submission",
        input_summary=f"candidate_id={result.get('candidate_id')}, decision={decision}",
        output_summary=f"status={status}, recommendation={result['recommendation']}",
        tool_used="streamlit_human_review",
        status="success",
    )
    return result


def _render_hr_review_brief(result: dict[str, Any], cv_text: str, job_text: str) -> None:
    """Show the evidence HR needs before making a manual decision."""
    skill_result = extract_skills(cv_text, job_text)
    review_reasons = result.get("review_reasons") or []
    tech = result.get("technical", {})
    profile = result.get("profile", {})

    st.markdown("#### HR Review Gate")
    st.caption("The AI is unsure, so HR must make the final hiring decision using the evidence below.")

    c1, c2, c3 = st.columns(3)
    c1.metric("Final score", f"{result.get('final_score', 0):.4f}", result.get("final_label", ""))
    c2.metric("Technical score", f"{tech.get('score', 0):.4f}", tech.get("label", ""))
    c3.metric("Profile score", f"{profile.get('score', 0):.4f}", profile.get("label", ""))

    if review_reasons:
        st.warning("Why HR review was triggered: " + ", ".join(review_reasons))

    left, right = st.columns(2)
    with left:
        st.markdown("**Matched job skills**")
        if skill_result.matched_skills:
            st.markdown(", ".join(skill_result.matched_skills))
        else:
            st.caption("No required skills were clearly matched.")

        st.markdown("**Technical agent evidence**")
        for item in tech.get("evidence", [])[:4]:
            st.markdown(f"- {item}")

    with right:
        st.markdown("**Missing job skills**")
        if skill_result.missing_skills:
            st.markdown(", ".join(skill_result.missing_skills))
        else:
            st.caption("No required skill gaps detected.")

        st.markdown("**Profile agent evidence**")
        evidence = profile.get("evidence", [])
        if evidence:
            for item in evidence[:4]:
                st.markdown(f"- {item}")
        else:
            st.caption("No strong profile evidence detected.")

    with st.expander("Agent reasoning", expanded=False):
        st.markdown(f"**Technical Matcher:** {tech.get('rationale', 'No rationale available.')}")
        st.markdown(f"**Profile Analyzer:** {profile.get('rationale', 'No rationale available.')}")
        st.markdown(f"**Orchestrator:** {result.get('orchestrator_summary', 'No summary available.')}")


def _load_confusion_snapshot(root: Path) -> dict[str, Any]:
    payload = _read_json_file(root / "models" / "model_evaluation.json") or {}
    matrix = payload.get("confusion_matrix") or []
    return {
        "accuracy": payload.get("accuracy", 0.0),
        "macro_f1": payload.get("macro_f1", 0.0),
        "confusion_matrix": matrix,
    }


def _read_audit_rows(log_file: Path, limit: int = 120) -> list[dict[str, Any]]:
    if not log_file.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in log_file.read_text(encoding="utf-8").splitlines()[-limit:]:
        try:
            rows.append(json.loads(line))
        except Exception:
            continue
    return rows


def _reason_list(result: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    reasons.extend(result.get("technical", {}).get("evidence", []))
    reasons.extend(result.get("profile", {}).get("evidence", []))
    if result.get("review_reasons"):
        reasons.extend([f"Review trigger: {item}" for item in result["review_reasons"]])
    # Keep only unique values while preserving order.
    seen: set[str] = set()
    deduped: list[str] = []
    for item in reasons:
        if item not in seen:
            seen.add(item)
            deduped.append(item)
    return deduped


def _build_defense_pack(
    *,
    result: dict[str, Any],
    candidate_id: str,
    job_id: str,
    cv_text: str,
    job_text: str,
    logger: JsonLogger,
    root: Path,
) -> dict[str, Any]:
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": "single-screening-defense",
        "candidate_id": candidate_id,
        "job_id": job_id,
        "result": result,
        "skill_analysis": {
            "matched": extract_skills(cv_text, job_text).matched_skills,
            "missing": extract_skills(cv_text, job_text).missing_skills,
        },
        "model_snapshot": _load_confusion_snapshot(root),
        "runtime_log_file": str(logger.log_file),
        "audit_tail": _read_audit_rows(logger.log_file, limit=30),
    }


def _render_batch_results(
    *,
    batch_results: list[dict[str, Any]],
    selected_job_id: str,
    job_text: str,
    generated_at: str,
    ollama_model_input: str,
    show_explain: bool,
    logger: JsonLogger,
    root: Path,
) -> None:
    st.success(f"Screened {len(batch_results)} candidates.")

    import pandas as pd

    label_emoji = {"High": "🟢", "Medium": "🟡", "Low": "🔴"}
    rows = []
    for rank, r in enumerate(batch_results, start=1):
        rows.append({
            "Rank": rank,
            "Candidate": r["candidate_id"],
            "Score": r["final_score"],
            "Label": f"{label_emoji.get(r['final_label'], '')} {r['final_label']}",
            "Recommendation": r["recommendation"],
            "Tech": round(r["technical"]["score"], 3),
            "Profile": round(r["profile"]["score"], 3),
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    chart_df = pd.DataFrame(
        {r["candidate_id"]: r["final_score"] for r in batch_results},
        index=["Score"],
    ).T
    st.bar_chart(chart_df, height=260)

    dl1, dl2 = st.columns(2)
    dl1.download_button(
        label="Download Batch Report (JSON)",
        data=json.dumps({"results": batch_results, "generated_at": generated_at}, indent=2),
        file_name=f"batch_report_{selected_job_id}.json",
        mime="application/json",
        use_container_width=True,
        on_click="ignore",
    )
    dl2.download_button(
        label="Download Batch Report (CSV)",
        data=_results_to_csv(batch_results),
        file_name=f"batch_report_{selected_job_id}.csv",
        mime="text/csv",
        use_container_width=True,
        on_click="ignore",
    )

    st.subheader("Candidate Analysis Summary")
    st.caption("Visible explanation for each ranked candidate. Open the full details below for tabs, audit logs, and raw agent outputs.")
    for r in batch_results:
        tech = r.get("technical", {})
        profile = r.get("profile", {})
        metadata = tech.get("metadata", {})
        matched = metadata.get("matched_skills", [])
        missing = metadata.get("missing_skills", [])
        review_reasons = r.get("review_reasons", [])
        human_review = r.get("human_review")

        with st.container(border=True):
            h1, h2, h3 = st.columns([2.2, 1, 1])
            h1.markdown(f"**#{r.get('batch_rank', '?')} {r['candidate_id']}**")
            h2.metric("Final score", f"{r['final_score']:.4f}", r["final_label"])
            h3.metric("Decision", r["recommendation"])

            s1, s2, s3 = st.columns(3)
            s1.markdown(f"**Technical Matcher:** {tech.get('label', 'n/a')} ({tech.get('score', 0):.4f})")
            s2.markdown(f"**Profile Analyzer:** {profile.get('label', 'n/a')} ({profile.get('score', 0):.4f})")
            s3.markdown(f"**Model:** {metadata.get('model_used', 'unknown')}")

            st.markdown(f"**Orchestrator:** {r.get('orchestrator_summary', 'No summary available.')}")
            st.markdown(f"**Technical rationale:** {tech.get('rationale', 'No rationale available.')}")
            st.markdown(f"**Profile rationale:** {profile.get('rationale', 'No rationale available.')}")

            e1, e2 = st.columns(2)
            e1.markdown("**Matched skills**")
            e1.write(", ".join(matched) if matched else "None clearly matched")
            e2.markdown("**Missing job skills**")
            e2.write(", ".join(missing) if missing else "None")

            if review_reasons:
                st.warning("Needs review because: " + ", ".join(review_reasons))
            if human_review:
                st.info(f"Human checkpoint status: {human_review.get('status')}")
            if r.get("llm_rationale"):
                st.info(f"Ollama rationale: {r['llm_rationale']}")

    with st.expander("Full individual result details"):
        for idx, r in enumerate(batch_results):
            with st.expander(f"{r['candidate_id']} - {r['final_label']} ({r['final_score']:.4f})", expanded=(idx == 0)):
                _render_single_result(
                    r,
                    ollama_model_input,
                    show_explain,
                    r.get("_cv_text", ""),
                    job_text,
                    logger,
                    root,
                )


# ── Result rendering ─────────────────────────────────────────────────────────

def _render_single_result(
    result: dict,
    ollama_model_input: str,
    show_explain: bool,
    cv_text: str,
    job_text: str,
    logger: JsonLogger,
    root: Path,
) -> None:
    label_colours = {"High": "#007a70", "Medium": "#e07b00", "Low": "#c0392b"}
    label = result["final_label"]
    colour = label_colours.get(label, "#333")
    disagreement = abs(result["technical"]["score"] - result["profile"]["score"])
    agreement_score = max(0.0, min(1.0, 1.0 - disagreement))
    model_source = result.get("technical", {}).get("metadata", {}).get("model_used", "unknown")
    skill_result = extract_skills(cv_text, job_text)

    tabs = st.tabs(["Intake", "Agent Analysis", "Decision Room", "Evidence", "Audit Trail"])

    with tabs[0]:
        st.subheader("Intake")
        i1, i2 = st.columns(2)
        i1.markdown(f"**Candidate ID:** {result.get('candidate_id')}")
        i1.markdown(f"**Job ID:** {result.get('job_id')}")
        i2.markdown(f"**Model source:** {model_source}")
        i2.markdown(f"**Ollama rationale enabled:** {'yes' if bool(result.get('llm_rationale')) else 'no'}")

        with st.expander("Candidate CV snapshot", expanded=False):
            st.text(cv_text[:3000])
        with st.expander("Job description snapshot", expanded=False):
            st.text(job_text[:3000])

    with tabs[1]:
        st.subheader("Agent Analysis")
        c1, c2, c3 = st.columns(3)
        c1.markdown(
            f"<div class='metric'><div class='metric-label'>Technical Score</div>"
            f"<div class='metric-value'>{result['technical']['score']:.4f}</div></div>",
            unsafe_allow_html=True,
        )
        c2.markdown(
            f"<div class='metric'><div class='metric-label'>Profile Score</div>"
            f"<div class='metric-value'>{result['profile']['score']:.4f}</div></div>",
            unsafe_allow_html=True,
        )
        c3.markdown(
            f"<div class='metric'><div class='metric-label'>Agent Agreement</div>"
            f"<div class='metric-value'>{agreement_score:.0%}</div></div>",
            unsafe_allow_html=True,
        )
        st.progress(agreement_score, text=f"Agreement meter (1 - |tech - profile|): {agreement_score:.0%}")

        import pandas as pd
        chart_data = pd.DataFrame(
            {"Score": [result["technical"]["score"], result["profile"]["score"], result["final_score"]]},
            index=["Technical", "Profile", "Overall"],
        )
        st.bar_chart(chart_data, height=220)

        lcol, rcol = st.columns(2)
        with lcol:
            st.subheader("Technical Matcher")
            st.write(result["technical"])
        with rcol:
            st.subheader("Profile Analyzer")
            st.write(result["profile"])

    with tabs[2]:
        st.subheader("Decision Room")
        d1, d2, d3 = st.columns(3)
        d1.markdown(
            f"<div class='metric'><div class='metric-label'>Final Label</div>"
            f"<div class='metric-value' style='color:{colour}'>{label}</div></div>",
            unsafe_allow_html=True,
        )
        d2.markdown(
            f"<div class='metric'><div class='metric-label'>Final Score</div>"
            f"<div class='metric-value'>{result['final_score']:.4f}</div></div>",
            unsafe_allow_html=True,
        )
        d3.markdown(
            f"<div class='metric'><div class='metric-label'>Recommendation</div>"
            f"<div class='metric-value'>{result['recommendation']}</div></div>",
            unsafe_allow_html=True,
        )

        st.markdown(f"**Orchestrator summary:** {result['orchestrator_summary']}")

        top_reasons = _reason_list(result)[:5]
        if top_reasons:
            st.markdown("**Top decision reasons**")
            for item in top_reasons:
                st.markdown(f"- {item}")

        if result.get("review_reasons"):
            plain = ", ".join(result["review_reasons"])
            st.warning(f"Human review trigger: {plain}")

        if result.get("llm_rationale"):
            st.info(f"Ollama ({ollama_model_input}) rationale: {result['llm_rationale']}")

    with tabs[3]:
        st.subheader("Evidence")
        e1, e2 = st.columns(2)
        with e1:
            st.markdown("**Matched skills**")
            if skill_result.matched_skills:
                st.markdown(
                    " ".join(f"<span class='skill-match'>✓ {s}</span>" for s in skill_result.matched_skills),
                    unsafe_allow_html=True,
                )
            else:
                st.caption("None")

        with e2:
            st.markdown("**Job skills not found in this CV**")
            missing = skill_result.missing_skills or []
            if missing:
                st.markdown(
                    " ".join(f"<span class='skill-miss'>✗ {s}</span>" for s in missing),
                    unsafe_allow_html=True,
                )
            else:
                st.caption("No skill gaps detected")

        st.progress(min(skill_result.coverage, 1.0), text=f"CV skill coverage: {skill_result.coverage:.0%}")

        total_req = len(skill_result.job_skills or [])
        missing_count = len(skill_result.missing_skills or [])
        matched_count = len(skill_result.matched_skills or [])
        import pandas as pd
        gap_df = pd.DataFrame(
            {"Count": [matched_count, missing_count, max(total_req - matched_count - missing_count, 0)]},
            index=["Matched", "Missing", "Other"],
        )
        st.bar_chart(gap_df, height=200)

        snapshot = _load_confusion_snapshot(root)
        st.markdown("**Model evaluation snapshot**")
        s1, s2 = st.columns(2)
        s1.metric("Model Accuracy", f"{float(snapshot.get('accuracy', 0.0)):.1%}")
        s2.metric("Macro F1", f"{float(snapshot.get('macro_f1', 0.0)):.1%}")
        st.caption("Confusion matrix from model evaluation artifact")
        st.json(snapshot.get("confusion_matrix", []))

        if show_explain:
            with st.expander("CV signal highlights", expanded=False):
                st.caption("Highlighted words indicate signal terms used by the pipeline features.")
                st.markdown(
                    f"<div style='line-height:1.9;font-size:0.95rem'>{_highlight_cv(cv_text, job_text)}</div>",
                    unsafe_allow_html=True,
                )

    with tabs[4]:
        st.subheader("Audit Trail")
        rows = _read_audit_rows(logger.log_file, limit=120)
        if not rows:
            st.info("No audit rows found yet.")
        else:
            st.caption(f"Showing last {len(rows)} events from {logger.log_file}")
            for row in rows[-30:][::-1]:
                status = str(row.get("status") or "unknown").upper()
                stamp = row.get("timestamp") or ""
                agent = row.get("agent_name") or "system"
                action = row.get("action") or row.get("event") or "event"
                st.markdown(
                    f"<div class='history-item'><b>{stamp}</b><br>{agent} · {action} · {status}</div>",
                    unsafe_allow_html=True,
                )

            st.download_button(
                label="Download Audit Tail (JSON)",
                data=json.dumps(rows, indent=2, ensure_ascii=False),
                file_name=f"audit_tail_{result.get('candidate_id')}_{result.get('job_id')}.json",
                mime="application/json",
                use_container_width=True,
            )

    with st.expander("Raw JSON output", expanded=False):
        st.json(result)


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    st.set_page_config(page_title="CV Screening Multi-Agent", page_icon="📄", layout="wide")
    _inject_styles()

    root = Path(__file__).resolve().parents[1]
    sample_cv, sample_job = _load_sample_data(root)

    st.markdown(
        """
        <div class="hero">
            <div class="hero-kicker">Recruiter Decision Cockpit</div>
            <h1>CV Screening Multi-Agent Dashboard</h1>
            <p>Run the specialist agents, inspect deep-learning evidence, capture human approval, and present a defense-ready hiring recommendation in one place.</p>
            <div class="hero-pills">
                <span class="hero-pill">2 Specialist Agents</span>
                <span class="hero-pill">PyTorch Tool in Workflow</span>
                <span class="hero-pill">Human Approval Gate</span>
                <span class="hero-pill">JSON Audit Trail</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    evidence = _collect_defense_evidence(root)
    _render_section_intro(
        "Defense Snapshot",
        "A compact overview of the model, pipeline, data coverage, and checklist evidence you can show in a live presentation.",
        eyebrow="Overview",
    )
    _render_showcase_cards(evidence)
    _render_brief_alignment(evidence)

    # ── Sidebar ───────────────────────────────────────────────────────────────
    st.sidebar.header("Mode")
    app_mode = st.sidebar.radio(
        "Application mode",
        ["Single Screening", "Batch Screening"],
        help=(
            "Single: screen one CV against one job.\n"
            "Batch: screen multiple CVs against one job, ranked leaderboard."
        ),
    )
    _render_quick_mode_guide()

    _render_runtime_status(root)

    st.sidebar.divider()
    st.sidebar.subheader("LLM Backend")
    use_ollama = st.sidebar.toggle("Enable Ollama", value=False, key="sidebar_ollama_toggle")
    ollama_model_input = settings.ollama_model
    ollama_status: str = ""
    st.sidebar.caption(f"Target endpoint: {settings.ollama_host}")

    probe_client = OllamaClient(host=settings.ollama_host, model=settings.ollama_model)
    if probe_client.is_available():
        st.sidebar.success("Ollama server is online")
    else:
        st.sidebar.warning("Ollama server is offline")
        if st.sidebar.button("Start Ollama Server", use_container_width=True, key="start_ollama_btn"):
            ok, msg = _start_ollama_server()
            if ok:
                st.sidebar.success(msg)
            else:
                st.sidebar.error(msg)

    if use_ollama:
        ollama_model_input = st.sidebar.text_input("Model", value=settings.ollama_model)
        _probe = OllamaClient(host=settings.ollama_host, model=ollama_model_input)
        if _probe.is_available():
            st.sidebar.success(f"Ollama running at {settings.ollama_host}")
            ollama_status = "available"
            if st.sidebar.button("Pull / Refresh Model", use_container_width=True, key="pull_ollama_model_btn"):
                with st.spinner(f"Pulling model {ollama_model_input}..."):
                    ok, msg = _pull_ollama_model(ollama_model_input)
                if ok:
                    st.sidebar.success(msg)
                else:
                    st.sidebar.error(msg)
        else:
            st.sidebar.warning(f"Ollama not detected at {settings.ollama_host}.\nStart it with: `ollama serve`")
            ollama_status = "unavailable"
    st.sidebar.caption("Ollama enriches results with an LLM rationale. It does not change agent scores.")

    st.sidebar.divider()
    st.sidebar.subheader("Options")
    show_explain = st.sidebar.toggle("Show explainability highlights", value=True, key="sidebar_explain_toggle")
    require_human_approval = st.sidebar.toggle(
        "Require human approval",
        value=True,
        help="When enabled, flagged decisions require explicit reviewer confirmation before final output is shown in single-screening mode.",
    )
    if require_human_approval and app_mode != "Single Screening":
        st.sidebar.caption("Strict blocking approval is applied in Single Screening mode. Batch mode marks pending review.")

    # ── Session history (shared across all modes) ─────────────────────────────
    if "history" not in st.session_state:
        st.session_state["history"] = []

    ollama_client: OllamaClient | None = None
    if use_ollama and ollama_status == "available":
        ollama_client = OllamaClient(host=settings.ollama_host, model=ollama_model_input)

    logger = JsonLogger(settings.log_dir_abs)

    # ═════════════════════════════════════════════════════════════════════════
    # MODE 1 — SINGLE SCREENING
    # ═════════════════════════════════════════════════════════════════════════
    if app_mode == "Single Screening":
        _render_section_intro(
            "Single Candidate Review",
            "Screen one CV against one role, inspect specialist outputs side by side, and capture reviewer approval when the system flags uncertainty.",
            eyebrow="Mode 1",
        )
        st.sidebar.divider()
        st.sidebar.subheader("Input Source")
        source = st.sidebar.radio("Choose source", ["Sample Data", "Manual Input", "Upload JSON", "Upload PDF"])
        defense_demo_mode = st.sidebar.toggle(
            "Defense demo mode",
            value=False,
            help="Shows architecture/checklist sections and enables one-click scripted sample run.",
        )

        cv_record: dict = sample_cv.copy()
        job_record: dict = sample_job.copy()
        cv_text_override: str | None = None

        if source == "Upload PDF":
            pdf_file = st.sidebar.file_uploader("Upload CV (PDF)", type=["pdf"])
            job_file_json = st.sidebar.file_uploader("Upload Job (JSON)", type=["json"])
            if pdf_file:
                cv_text_override = _extract_pdf_text(pdf_file.getvalue())
            loaded_job = _load_json_from_upload(job_file_json)
            if job_file_json is not None and loaded_job is None:
                st.sidebar.error("Invalid Job JSON format.")
            if loaded_job:
                job_record = loaded_job if isinstance(loaded_job, dict) else loaded_job[0]
        elif source == "Upload JSON":
            cv_file_json = st.sidebar.file_uploader("Upload CV JSON", type=["json"])
            job_file_json = st.sidebar.file_uploader("Upload Job JSON", type=["json"])
            loaded_cv = _load_json_from_upload(cv_file_json)
            loaded_job = _load_json_from_upload(job_file_json)
            if cv_file_json is not None and loaded_cv is None:
                st.sidebar.error("Invalid CV JSON format.")
            if job_file_json is not None and loaded_job is None:
                st.sidebar.error("Invalid Job JSON format.")
            if loaded_cv:
                cv_record = loaded_cv if isinstance(loaded_cv, dict) else loaded_cv[0]
            if loaded_job:
                job_record = loaded_job if isinstance(loaded_job, dict) else loaded_job[0]
        elif source == "Manual Input":
            cv_record["candidate_id"] = st.sidebar.text_input("Candidate ID", value=sample_cv["candidate_id"])
            job_record["job_id"] = st.sidebar.text_input("Job ID", value=sample_job["job_id"])

        left, right = st.columns(2)
        with left:
            cv_text = _as_text(st.text_area("Candidate CV", value=cv_text_override or cv_record.get("cv_text", ""), height=220))
        with right:
            job_text = _as_text(st.text_area("Job Description", value=job_record.get("job_text", ""), height=220))

        candidate_id = _as_text(st.text_input("Candidate ID", value=cv_record.get("candidate_id", "cand_demo")))
        job_id = _as_text(st.text_input("Job ID", value=job_record.get("job_id", "job_demo")))

        if defense_demo_mode:
            with st.expander("Architecture summary", expanded=True):
                st.markdown("- Orchestrator combines Technical Matcher and Profile Analyzer.")
                st.markdown("- Technical Matcher uses DL model tool and skill extractor.")
                st.markdown("- Borderline or conflicted outcomes route to HITL checkpoint.")
            with st.expander("Requirement checklist", expanded=True):
                st.markdown("- Multi-agent roles: PASS")
                st.markdown("- DL model as callable tool: PASS")
                st.markdown("- HITL decision gate: PASS")
                st.markdown("- JSON logging and audit trail: PASS")
                st.markdown("- Reproducible run/evaluation commands: PASS")

        run_col, demo_col = st.columns(2)
        run_clicked = run_col.button("Run Screening", type="primary", use_container_width=True)
        demo_clicked = demo_col.button(
            "Run Defense Demo",
            use_container_width=True,
            disabled=not defense_demo_mode,
            help="Runs a scripted sample with export-ready evidence.",
        )

        if demo_clicked:
            candidate_id = _as_text(sample_cv.get("candidate_id", candidate_id), candidate_id)
            job_id = _as_text(sample_job.get("job_id", job_id), job_id)
            cv_text = _as_text(sample_cv.get("cv_text", cv_text), cv_text)
            job_text = _as_text(sample_job.get("job_text", job_text), job_text)

        if not run_clicked and not demo_clicked:
            st.info("Fill the inputs and click Run Screening.")
            _render_history()
            return

        if not cv_text.strip() or not job_text.strip():
            st.error("Both CV and Job Description must be provided.")
            return

        with st.spinner("Running agents" + (" + Ollama..." if ollama_client else "...")):
            result = _run_one(
                cv_text,
                job_text,
                candidate_id,
                job_id,
                logger,
                ollama_client,
                require_human_approval=require_human_approval,
            )

        review = result.get("human_review") or {}
        review_status = review.get("status")
        if require_human_approval and review_status == "pending-human-approval":
            st.warning("HR review is required before the final recommendation can be finalized.")
            _render_hr_review_brief(result, cv_text, job_text)
            decision = st.radio(
                "Final HR decision",
                ["Shortlist", "Reject", "Needs Review"],
                horizontal=True,
                key=f"decision_{candidate_id}_{job_id}",
            )
            reviewer = st.text_input(
                "Reviewer name",
                value="",
                key=f"reviewer_{candidate_id}_{job_id}",
            )
            reviewer_comment = st.text_area(
                "Reviewer comment (optional)",
                value="",
                key=f"reviewer_comment_{candidate_id}_{job_id}",
                height=80,
            )
            if not st.button("Confirm HR Decision", type="primary", use_container_width=True):
                st.info("Review the analysis above, then confirm the HR decision to continue.")
                st.stop()

            result = _apply_manual_human_decision(result, logger, decision, reviewer, reviewer_comment)

        st.session_state["history"].insert(0, {
            "ts": datetime.now(timezone.utc).strftime("%H:%M:%S"),
            "candidate_id": candidate_id,
            "job_id": job_id,
            "label": result["final_label"],
            "score": result["final_score"],
            "recommendation": result["recommendation"],
        })

        _render_single_result(result, ollama_model_input, show_explain, cv_text, job_text, logger, root)

        report_payload = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "source": "cv-screening-multiagent-streamlit",
            "result": result,
        }
        defense_pack = _build_defense_pack(
            result=result,
            candidate_id=candidate_id,
            job_id=job_id,
            cv_text=cv_text,
            job_text=job_text,
            logger=logger,
            root=root,
        )
        report_name = f"screening_report_{candidate_id}_{job_id}.json".replace(" ", "_")
        dl1, dl2, dl3 = st.columns(3)
        dl1.download_button(
            label="Download Report (JSON)",
            data=json.dumps(report_payload, indent=2, ensure_ascii=False),
            file_name=report_name,
            mime="application/json",
            use_container_width=True,
        )
        dl2.download_button(
            label="Download Report (CSV)",
            data=_results_to_csv([result]),
            file_name=report_name.replace(".json", ".csv"),
            mime="text/csv",
            use_container_width=True,
        )
        dl3.download_button(
            label="Download Defense Pack (JSON)",
            data=json.dumps(defense_pack, indent=2, ensure_ascii=False),
            file_name=report_name.replace(".json", "_defense_pack.json"),
            mime="application/json",
            use_container_width=True,
        )

        st.caption(f"Log file: {logger.log_file}")
        _render_history()

    # ═════════════════════════════════════════════════════════════════════════
    # MODE 2 — BATCH SCREENING
    # ═════════════════════════════════════════════════════════════════════════
    elif app_mode == "Batch Screening":
        _render_section_intro(
            "Batch Candidate Leaderboard",
            "Rank multiple candidates against one target role and export a shortlist-ready leaderboard with underlying agent scores.",
            eyebrow="Mode 2",
        )
        st.subheader("Batch Screening — candidate leaderboard")
        st.caption("Screen multiple CVs against a single job and rank by fit score.")

        batch_source = st.radio(
            "CV source",
            ["Sample CVs (built-in)", "Upload CV list (JSON)", "Upload CV PDFs"],
            horizontal=True,
        )
        cvs: list[dict] = _load_all_cvs(root)

        if batch_source == "Upload CV list (JSON)":
            uploaded_cvs = st.file_uploader("Upload CVs JSON (list)", type=["json"])
            if uploaded_cvs:
                loaded = _load_json_from_upload(uploaded_cvs)
                if isinstance(loaded, list):
                    cvs = loaded
                elif isinstance(loaded, dict):
                    cvs = [loaded]
                else:
                    st.error("Invalid CV list JSON format.")
        elif batch_source == "Upload CV PDFs":
            uploaded_pdfs = st.file_uploader(
                "Upload one or more CV PDFs",
                type=["pdf"],
                accept_multiple_files=True,
            )
            if uploaded_pdfs:
                cvs = _load_cvs_from_pdf_uploads(uploaded_pdfs)
                with st.expander("Extracted PDF CVs", expanded=False):
                    for cv in cvs:
                        st.markdown(f"**{cv['candidate_id']}** from `{cv.get('source_file', '')}`")
                        if cv.get("extraction_status") != "success":
                            st.error("Text extraction failed or produced empty text for this PDF.")
                        st.text(_as_text(cv.get("cv_text", ""))[:1500])
            else:
                cvs = []

        st.markdown(f"**{len(cvs)} CV(s) loaded.**")

        jobs = _load_all_jobs(root)
        job_options = {j["job_id"]: j for j in jobs}
        selected_job_id = _as_text(st.selectbox("Select job", list(job_options.keys())))
        selected_job = job_options[selected_job_id]

        job_text = _as_text(st.text_area("Job description", value=selected_job["job_text"], height=120))

        run_batch = st.button("Run Batch Screening", type="primary", use_container_width=True)
        if not run_batch:
            saved_batch = st.session_state.get("last_batch_screening")
            if (
                saved_batch
                and saved_batch.get("job_id") == selected_job_id
                and saved_batch.get("job_text") == job_text
            ):
                st.info("Showing the latest batch analysis. Click Run Batch Screening to refresh it.")
                _render_batch_results(
                    batch_results=saved_batch["results"],
                    selected_job_id=saved_batch["job_id"],
                    job_text=saved_batch["job_text"],
                    generated_at=saved_batch["generated_at"],
                    ollama_model_input=ollama_model_input,
                    show_explain=show_explain,
                    logger=logger,
                    root=root,
                )
            elif saved_batch:
                st.info("Inputs changed. Click Run Batch Screening to analyze this job/CV set.")
            else:
                st.info("Select a job and click Run Batch Screening.")
            _render_history()
            return

        valid_cvs = _ensure_unique_candidate_ids(
            [
                cv for cv in cvs
                if _as_text(cv.get("cv_text", "")).strip()
                and cv.get("extraction_status", "success") == "success"
            ]
        )
        if not valid_cvs:
            st.error("At least one CV with cv_text is required.")
            return

        batch_results: list[dict] = []
        progress = st.progress(0, text="Starting…")
        raw_results = run_batch_screening(
            cv_records=valid_cvs,
            job_record={"job_id": selected_job_id, "job_text": job_text},
            model_path=settings.model_path_abs,
            low=settings.borderline_low,
            high=settings.borderline_high,
            logger=logger,
            ollama_client=ollama_client,
            require_human_approval=require_human_approval,
        )
        cv_lookup = {cv["candidate_id"]: _as_text(cv.get("cv_text", "")) for cv in valid_cvs}
        for idx, r in enumerate(raw_results):
            progress.progress((idx + 1) / len(raw_results), text=f"Completed {r.get('candidate_id', idx + 1)}")
            r["_cv_text"] = cv_lookup.get(r["candidate_id"], "")
            batch_results.append(r)
            st.session_state["history"].insert(0, {
                "ts": datetime.now(timezone.utc).strftime("%H:%M:%S"),
                "candidate_id": r["candidate_id"],
                "job_id": r["job_id"],
                "label": r["final_label"],
                "score": r["final_score"],
                "recommendation": r["recommendation"],
            })
        progress.empty()

        generated_at = datetime.now(timezone.utc).isoformat()
        st.session_state["last_batch_screening"] = {
            "results": batch_results,
            "job_id": selected_job_id,
            "job_text": job_text,
            "generated_at": generated_at,
        }
        _render_batch_results(
            batch_results=batch_results,
            selected_job_id=selected_job_id,
            job_text=job_text,
            generated_at=generated_at,
            ollama_model_input=ollama_model_input,
            show_explain=show_explain,
            logger=logger,
            root=root,
        )

        _render_history()


# ── Session history renderer ─────────────────────────────────────────────────

def _render_history() -> None:
    history = st.session_state.get("history", [])
    if not history:
        return
    with st.sidebar.expander(f"Session history ({len(history)})", expanded=False):
        for h in history[:20]:
            label_colour = {"High": "#007a70", "Medium": "#e07b00", "Low": "#c0392b"}.get(h["label"], "#333")
            st.markdown(
                f"<div class='history-item'>"
                f"<b>{h['candidate_id']}</b> × {h['job_id']}<br>"
                f"<span style='color:{label_colour};font-weight:700'>{h['label']}</span> "
                f"({h['score']:.4f}) · {h['ts']}</div>",
                unsafe_allow_html=True,
            )


if __name__ == "__main__":
    main()
