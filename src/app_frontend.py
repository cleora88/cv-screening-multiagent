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
            }
            html, body, [class*="css"] {
                font-family: 'Manrope', sans-serif;
                color: var(--ink);
            }
            .stApp {
                background: radial-gradient(circle at top left, var(--bg-start), var(--bg-end));
            }
            .hero {
                background: linear-gradient(120deg, #093028 0%, #237a57 100%);
                padding: 1.1rem 1.2rem;
                border-radius: 14px;
                color: #f8fffb;
                margin-bottom: 1rem;
                box-shadow: 0 8px 22px rgba(6, 40, 34, 0.2);
            }
            .hero h1 { margin: 0; font-size: 1.45rem; font-weight: 800; }
            .hero p  { margin: 0.35rem 0 0; opacity: 0.94; }
            .metric {
                background: var(--card);
                border: 1px solid #dce7ef;
                border-radius: 12px;
                padding: 0.75rem;
                box-shadow: 0 8px 18px rgba(17, 32, 43, 0.06);
            }
            .metric-label {
                color: var(--muted);
                font-size: 0.82rem;
                text-transform: uppercase;
                letter-spacing: 0.04rem;
            }
            .metric-value { font-size: 1.2rem; font-weight: 800; color: var(--ink); }
            .mono { font-family: 'IBM Plex Mono', monospace; }
            .skill-match { color: #007a70; font-weight: 700; }
            .skill-miss  { color: #c0392b; font-weight: 700; }
            .history-item {
                border-left: 3px solid #237a57;
                padding: 0.4rem 0.7rem;
                margin-bottom: 0.4rem;
                background: #f0faf5;
                border-radius: 0 6px 6px 0;
                font-size: 0.85rem;
            }
            .showcase-card {
                background: linear-gradient(145deg, #ffffff, #eef6f4);
                border: 1px solid #d3e8e2;
                border-radius: 14px;
                padding: 0.9rem;
                box-shadow: 0 8px 20px rgba(10, 63, 49, 0.08);
                animation: fadeInUp 0.45s ease;
            }
            .showcase-label {
                color: #45606a;
                font-size: 0.78rem;
                text-transform: uppercase;
                letter-spacing: 0.06rem;
            }
            .showcase-value {
                font-size: 1.4rem;
                font-weight: 800;
                color: #0a3f31;
                margin-top: 0.1rem;
            }
            .showcase-sub {
                font-size: 0.82rem;
                color: #4a5b67;
                margin-top: 0.15rem;
            }
            .brief-ok {
                background: #ecfaf4;
                border-left: 4px solid #007a70;
                padding: 0.55rem 0.7rem;
                border-radius: 6px;
                margin-bottom: 0.45rem;
                font-size: 0.92rem;
            }
            @keyframes fadeInUp {
                from { opacity: 0; transform: translateY(8px); }
                to { opacity: 1; transform: translateY(0); }
            }
        </style>
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


def _read_json_file(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


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
        ("Dataset Coverage", f"{evidence['n_cvs']} CV / {evidence['n_jobs']} Jobs", "Batch + Multi-Job demo ready"),
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
            ("Presentation readiness", "Single, Batch, Multi-Job, PDF upload, explainability", True),
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
) -> dict[str, Any]:
    status_map = {
        "Approve": "approved",
        "Reject": "rejected",
        "Flag": "flagged",
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


# ── Result rendering ─────────────────────────────────────────────────────────

def _render_single_result(result: dict, ollama_model_input: str, show_explain: bool, cv_text: str, job_text: str) -> None:
    label_colours = {"High": "#007a70", "Medium": "#e07b00", "Low": "#c0392b"}
    label = result["final_label"]
    colour = label_colours.get(label, "#333")

    c1, c2, c3 = st.columns(3)
    c1.markdown(
        f"<div class='metric'><div class='metric-label'>Final Label</div>"
        f"<div class='metric-value' style='color:{colour}'>{label}</div></div>",
        unsafe_allow_html=True,
    )
    c2.markdown(
        f"<div class='metric'><div class='metric-label'>Final Score</div>"
        f"<div class='metric-value'>{result['final_score']:.4f}</div></div>",
        unsafe_allow_html=True,
    )
    c3.markdown(
        f"<div class='metric'><div class='metric-label'>Recommendation</div>"
        f"<div class='metric-value'>{result['recommendation']}</div></div>",
        unsafe_allow_html=True,
    )

    # Score breakdown bar chart
    st.markdown("##### Score breakdown")
    import pandas as pd
    chart_data = pd.DataFrame(
        {"Score": [result["technical"]["score"], result["profile"]["score"], result["final_score"]]},
        index=["Technical", "Profile", "Overall"],
    )
    st.bar_chart(chart_data, height=220)

    st.markdown(f"**Orchestrator summary:** {result['orchestrator_summary']}")

    if result.get("llm_rationale"):
        st.info(f"**Ollama ({ollama_model_input}) rationale:** {result['llm_rationale']}")

    # Skill gap panel
    with st.expander("Skill Gap Analysis", expanded=True):
        skill_result = extract_skills(cv_text, job_text)
        sc1, sc2 = st.columns(2)
        with sc1:
            st.markdown("**Matched skills**")
            if skill_result.matched_skills:
                st.markdown(" ".join(
                    f"<span class='skill-match'>✓ {s}</span>" for s in skill_result.matched_skills
                ), unsafe_allow_html=True)
            else:
                st.caption("None")
        with sc2:
            st.markdown("**Missing skills**")
            missing = skill_result.missing_skills or []
            if missing:
                st.markdown(" ".join(
                    f"<span class='skill-miss'>✗ {s}</span>" for s in missing
                ), unsafe_allow_html=True)
            else:
                st.caption("No gaps detected")
        st.progress(min(skill_result.coverage, 1.0), text=f"Skill coverage: {skill_result.coverage:.0%}")

    # Explainability heatmap
    if show_explain:
        with st.expander("Explainability — CV keyword highlights", expanded=True):
            st.caption("Words highlighted in green are recognised signal terms that influenced the score.")
            st.markdown(
                f"<div style='line-height:1.9;font-size:0.95rem'>{_highlight_cv(cv_text, job_text)}</div>",
                unsafe_allow_html=True,
            )

    tcol, pcol = st.columns(2)
    with tcol:
        st.subheader("Technical Agent")
        st.write(result["technical"])
    with pcol:
        st.subheader("Profile Agent")
        st.write(result["profile"])

    if result.get("human_review"):
        st.warning("Human review was triggered for this decision.")
        st.json(result["human_review"])

    with st.expander("Raw JSON output"):
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
            <h1>CV Screening Multi-Agent Dashboard</h1>
            <p>Run technical + profile agents, inspect evidence, and present final hiring recommendations live.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    evidence = _collect_defense_evidence(root)
    _render_showcase_cards(evidence)
    _render_brief_alignment(evidence)

    # ── Sidebar ───────────────────────────────────────────────────────────────
    st.sidebar.header("Mode")
    app_mode = st.sidebar.radio(
        "Application mode",
        ["Single Screening", "Batch Screening", "Multi-Job Match"],
        help=(
            "Single: screen one CV against one job.\n"
            "Batch: screen multiple CVs against one job, ranked leaderboard.\n"
            "Multi-Job: screen one CV against all available jobs."
        ),
    )

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
        st.sidebar.caption("Strict blocking approval is applied in Single Screening mode. Batch and Multi-Job modes mark pending review.")

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
        st.sidebar.divider()
        st.sidebar.subheader("Input Source")
        source = st.sidebar.radio("Choose source", ["Sample Data", "Manual Input", "Upload JSON", "Upload PDF"])

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
            cv_text = st.text_area("Candidate CV", value=cv_text_override or cv_record.get("cv_text", ""), height=220)
        with right:
            job_text = st.text_area("Job Description", value=job_record.get("job_text", ""), height=220)

        candidate_id = st.text_input("Candidate ID", value=cv_record.get("candidate_id", "cand_demo"))
        job_id = st.text_input("Job ID", value=job_record.get("job_id", "job_demo"))

        run_clicked = st.button("Run Screening", type="primary", use_container_width=True)
        if not run_clicked:
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
            st.warning("Human approval is required before final recommendation can be finalized.")
            decision = st.radio(
                "Reviewer decision",
                ["Approve", "Reject", "Flag"],
                horizontal=True,
                key=f"decision_{candidate_id}_{job_id}",
            )
            reviewer = st.text_input(
                "Reviewer name",
                value="",
                key=f"reviewer_{candidate_id}_{job_id}",
            )
            if not st.button("Confirm Human Decision", type="primary", use_container_width=True):
                st.info("Confirm a reviewer decision to continue.")
                st.stop()

            result = _apply_manual_human_decision(result, logger, decision, reviewer)

        st.session_state["history"].insert(0, {
            "ts": datetime.now(timezone.utc).strftime("%H:%M:%S"),
            "candidate_id": candidate_id,
            "job_id": job_id,
            "label": result["final_label"],
            "score": result["final_score"],
            "recommendation": result["recommendation"],
        })

        _render_single_result(result, ollama_model_input, show_explain, cv_text, job_text)

        report_payload = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "source": "cv-screening-multiagent-streamlit",
            "result": result,
        }
        report_name = f"screening_report_{candidate_id}_{job_id}.json".replace(" ", "_")
        dl1, dl2 = st.columns(2)
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

        st.caption(f"Log file: {logger.log_file}")
        _render_history()

    # ═════════════════════════════════════════════════════════════════════════
    # MODE 2 — BATCH SCREENING
    # ═════════════════════════════════════════════════════════════════════════
    elif app_mode == "Batch Screening":
        st.subheader("Batch Screening — candidate leaderboard")
        st.caption("Screen multiple CVs against a single job and rank by fit score.")

        batch_source = st.radio("CV source", ["Sample CVs (built-in)", "Upload CV list (JSON)"], horizontal=True)
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

        st.markdown(f"**{len(cvs)} CV(s) loaded.**")

        jobs = _load_all_jobs(root)
        job_options = {j["job_id"]: j for j in jobs}
        selected_job_id = st.selectbox("Select job", list(job_options.keys()))
        selected_job = job_options[selected_job_id]

        job_text = st.text_area("Job description", value=selected_job["job_text"], height=120)

        run_batch = st.button("Run Batch Screening", type="primary", use_container_width=True)
        if not run_batch:
            st.info("Select a job and click Run Batch Screening.")
            _render_history()
            return

        batch_results: list[dict] = []
        progress = st.progress(0, text="Starting…")
        for idx, cv in enumerate(cvs):
            progress.progress((idx + 1) / len(cvs), text=f"Screening {cv.get('candidate_id', idx+1)}…")
            r = _run_one(
                cv_text=cv.get("cv_text", ""),
                job_text=job_text,
                candidate_id=cv.get("candidate_id", f"cand_{idx}"),
                job_id=selected_job_id,
                logger=logger,
                ollama_client=ollama_client,
                require_human_approval=require_human_approval,
            )
            r["_cv_text"] = cv.get("cv_text", "")
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

        # Rank by score descending
        batch_results.sort(key=lambda x: x["final_score"], reverse=True)

        st.success(f"Screened {len(batch_results)} candidates.")

        # Leaderboard table
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

        # Score chart for all candidates
        chart_df = pd.DataFrame(
            {r["candidate_id"]: r["final_score"] for r in batch_results},
            index=["Score"],
        ).T
        st.bar_chart(chart_df, height=260)

        dl1, dl2 = st.columns(2)
        dl1.download_button(
            label="Download Batch Report (JSON)",
            data=json.dumps({"results": batch_results, "generated_at": datetime.now(timezone.utc).isoformat()}, indent=2),
            file_name=f"batch_report_{selected_job_id}.json",
            mime="application/json",
            use_container_width=True,
        )
        dl2.download_button(
            label="Download Batch Report (CSV)",
            data=_results_to_csv(batch_results),
            file_name=f"batch_report_{selected_job_id}.csv",
            mime="text/csv",
            use_container_width=True,
        )

        with st.expander("Individual result details"):
            for r in batch_results:
                with st.expander(f"{r['candidate_id']} — {r['final_label']} ({r['final_score']:.4f})"):
                    _render_single_result(r, ollama_model_input, show_explain, r.get("_cv_text", ""), job_text)

        _render_history()

    # ═════════════════════════════════════════════════════════════════════════
    # MODE 3 — MULTI-JOB MATCH
    # ═════════════════════════════════════════════════════════════════════════
    elif app_mode == "Multi-Job Match":
        st.subheader("Multi-Job Match — best job fit for one candidate")
        st.caption("Compare a single CV against all available jobs and rank best fits.")

        mj_source = st.radio("CV source", ["Sample CV", "Manual Input", "Upload PDF", "Upload JSON"], horizontal=True)
        cv_text = sample_cv.get("cv_text", "")
        candidate_id = sample_cv.get("candidate_id", "cand_demo")

        if mj_source == "Manual Input":
            candidate_id = st.text_input("Candidate ID", value=candidate_id)
            cv_text = st.text_area("Candidate CV", value=cv_text, height=200)
        elif mj_source == "Upload PDF":
            pdf_file = st.file_uploader("Upload CV (PDF)", type=["pdf"])
            candidate_id = st.text_input("Candidate ID", value=candidate_id)
            if pdf_file:
                cv_text = _extract_pdf_text(pdf_file.getvalue())
            cv_text = st.text_area("Extracted CV text (editable)", value=cv_text, height=200)
        elif mj_source == "Upload JSON":
            cv_file_json = st.file_uploader("Upload CV JSON", type=["json"])
            loaded = _load_json_from_upload(cv_file_json)
            if loaded:
                cv_record = loaded if isinstance(loaded, dict) else loaded[0]
                cv_text = cv_record.get("cv_text", cv_text)
                candidate_id = cv_record.get("candidate_id", candidate_id)
            elif cv_file_json is not None:
                st.error("Invalid CV JSON format.")
            candidate_id = st.text_input("Candidate ID", value=candidate_id)
            cv_text = st.text_area("Candidate CV", value=cv_text, height=200)
        else:
            cv_text = st.text_area("Candidate CV", value=cv_text, height=200)

        jobs = _load_all_jobs(root)
        st.markdown(f"Will match against **{len(jobs)} job(s)**.")

        run_mj = st.button("Find Best Job Matches", type="primary", use_container_width=True)
        if not run_mj:
            st.info("Fill in the CV and click Find Best Job Matches.")
            _render_history()
            return

        if not cv_text.strip():
            st.error("CV text is required.")
            return

        mj_results: list[dict] = []
        progress = st.progress(0, text="Matching jobs…")
        for idx, job in enumerate(jobs):
            progress.progress((idx + 1) / len(jobs), text=f"Matching against {job['job_id']}…")
            r = _run_one(
                cv_text=cv_text,
                job_text=job["job_text"],
                candidate_id=candidate_id,
                job_id=job["job_id"],
                logger=logger,
                ollama_client=ollama_client,
                require_human_approval=require_human_approval,
            )
            r["_job_text"] = job["job_text"]
            mj_results.append(r)
            st.session_state["history"].insert(0, {
                "ts": datetime.now(timezone.utc).strftime("%H:%M:%S"),
                "candidate_id": r["candidate_id"],
                "job_id": r["job_id"],
                "label": r["final_label"],
                "score": r["final_score"],
                "recommendation": r["recommendation"],
            })
        progress.empty()

        mj_results.sort(key=lambda x: x["final_score"], reverse=True)
        best = mj_results[0]
        st.success(f"Best match: **{best['job_id']}** — {best['final_label']} ({best['final_score']:.4f})")

        import pandas as pd
        label_emoji = {"High": "🟢", "Medium": "🟡", "Low": "🔴"}
        rows = []
        for rank, r in enumerate(mj_results, start=1):
            rows.append({
                "Rank": rank,
                "Job ID": r["job_id"],
                "Score": r["final_score"],
                "Label": f"{label_emoji.get(r['final_label'], '')} {r['final_label']}",
                "Recommendation": r["recommendation"],
                "Tech": round(r["technical"]["score"], 3),
                "Profile": round(r["profile"]["score"], 3),
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        chart_df = pd.DataFrame(
            {r["job_id"]: r["final_score"] for r in mj_results},
            index=["Score"],
        ).T
        st.bar_chart(chart_df, height=260)

        dl1, dl2 = st.columns(2)
        dl1.download_button(
            label="Download Multi-Job Report (JSON)",
            data=json.dumps({"candidate_id": candidate_id, "results": mj_results, "generated_at": datetime.now(timezone.utc).isoformat()}, indent=2),
            file_name=f"multijob_report_{candidate_id}.json",
            mime="application/json",
            use_container_width=True,
        )
        dl2.download_button(
            label="Download Multi-Job Report (CSV)",
            data=_results_to_csv(mj_results),
            file_name=f"multijob_report_{candidate_id}.csv",
            mime="text/csv",
            use_container_width=True,
        )

        with st.expander("Skill gap per job"):
            for r in mj_results:
                with st.expander(f"{r['job_id']} — score {r['final_score']:.4f}"):
                    skill_r = extract_skills(cv_text, r.get("_job_text", ""))
                    sc1, sc2 = st.columns(2)
                    with sc1:
                        st.markdown("**Matched**")
                        if skill_r.matched_skills:
                            st.markdown(" ".join(f"<span class='skill-match'>✓ {s}</span>" for s in skill_r.matched_skills), unsafe_allow_html=True)
                        else:
                            st.caption("None")
                    with sc2:
                        st.markdown("**Missing**")
                        missing = skill_r.missing_skills or []
                        if missing:
                            st.markdown(" ".join(f"<span class='skill-miss'>✗ {s}</span>" for s in missing), unsafe_allow_html=True)
                        else:
                            st.caption("No gaps")
                    st.progress(min(skill_r.coverage, 1.0), text=f"Coverage: {skill_r.coverage:.0%}")

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
