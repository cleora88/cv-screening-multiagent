from __future__ import annotations

from pathlib import Path


def _wrap_text(line: str, width: int = 92) -> list[str]:
    if len(line) <= width:
        return [line]
    words = line.split()
    out: list[str] = []
    chunk = ""
    for word in words:
        candidate = f"{chunk} {word}".strip()
        if len(candidate) <= width:
            chunk = candidate
        else:
            out.append(chunk)
            chunk = word
    if chunk:
        out.append(chunk)
    return out


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    report_md = root / "REPORT.md"
    out_pdf = root / "submission" / "final_report.pdf"
    out_pdf.parent.mkdir(parents=True, exist_ok=True)

    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.pdfgen import canvas
    except Exception as exc:  # pragma: no cover
        raise SystemExit(
            "reportlab is required to export PDF. Install dependencies with: python -m pip install -r requirements.txt"
        ) from exc

    lines = report_md.read_text(encoding="utf-8").splitlines()

    appendix_lines = [
        "",
        "## Appendix A - Reproducibility Checklist",
        "",
        "1. Python version and environment details",
        "- Recommended local runtime: Python 3.11/3.12 for CrewAI path.",
        "- Deterministic runtime is available for wider compatibility.",
        "",
        "2. Setup commands",
        "- pip install -r requirements.txt",
        "- python -m src.train_baseline",
        "- python -m src.main --runtime auto",
        "- python -m src.evaluate_pipeline",
        "",
        "3. Strict HITL validation",
        "- python -m src.main --runtime deterministic --require-human-approval",
        "- Expected behavior: pending review in non-interactive contexts and explicit review capture in frontend single mode.",
        "",
        "4. Output artifacts",
        "- models/cv_fit_model.pt",
        "- models/model_evaluation.json",
        "- logs/pipeline_evaluation.json",
        "- logs/run_*.jsonl",
        "",
        "## Appendix B - Logging Contract",
        "",
        "Each runtime log event stores the following fields:",
        "- timestamp",
        "- agent_name",
        "- action",
        "- tool_used",
        "- input_summary",
        "- output_summary",
        "- status",
        "- error",
        "- event",
        "- payload",
        "",
        "This schema enables consistent debugging, failure triage, and oral-defense traceability.",
        "",
        "## Appendix C - Guardrails and Failure Behavior",
        "",
        "Guardrails:",
        "- Borderline score threshold policy",
        "- Specialist disagreement threshold",
        "- Human checkpoint for sensitive decisions",
        "- Fallback behavior for tool and LLM failures",
        "",
        "Failure handling examples:",
        "- Invalid JSON/file type in CLI: graceful error + hint output",
        "- Missing model or load error: deterministic fallback prediction",
        "- Ollama failure: runtime warning and non-crashing continuation",
        "",
        "## Appendix D - Defense Notes",
        "",
        "Key design rationale to defend:",
        "- Why technical and profile signals are separated into specialists",
        "- Why orchestrator weighting and review thresholds exist",
        "- Why strict HITL is needed for high-impact screening decisions",
        "- Why structured JSON logging is essential for auditability",
        "",
        "Known limitations and roadmap:",
        "- Dataset scale and realism expansion",
        "- Semantic matching and fairness diagnostics",
        "- Batch-level reviewer workflow enhancements",
        "",
        "## Appendix E - Example Defense Q and A",
        "",
        "Q1: Why separate technical and profile analysis?",
        "A1: Separation improves interpretability, enables targeted evidence, and avoids collapsing heterogeneous signals into one opaque score.",
        "",
        "Q2: Why does the orchestrator keep weighted scoring?",
        "A2: Weighted aggregation provides deterministic governance and can be defended with explicit trade-offs between technical fit and profile fit.",
        "",
        "Q3: Why require HITL for borderline decisions?",
        "A3: Borderline and conflict cases have higher decision uncertainty; a reviewer checkpoint prevents automated overreach.",
        "",
        "Q4: How is model integration meaningful?",
        "A4: The technical specialist directly uses model output in scoring, and model confidence/evidence is propagated to the final decision context.",
        "",
        "Q5: How are failures prevented from crashing the pipeline?",
        "A5: Tool-level try/except guards, fallback outputs, and orchestrator review escalation keep runs resilient.",
        "",
        "Q6: What proves reproducibility?",
        "A6: Stable commands for setup, training, evaluation, runtime execution, and persistent JSONL traces for replay and audit.",
        "",
        "Q7: Why keep CrewAI with deterministic fallback?",
        "A7: CrewAI fulfills framework requirements while fallback preserves continuity in environments where optional dependencies are unavailable.",
        "",
        "Q8: What are next engineering priorities?",
        "A8: Larger datasets, semantic retrieval for skill matching, stricter schema validation, and enhanced reviewer workflow analytics.",
    ]
    lines.extend(appendix_lines)
    c = canvas.Canvas(str(out_pdf), pagesize=A4)
    width, height = A4

    left_margin = 48
    top_margin = 52
    line_height = 14
    y = height - top_margin

    c.setTitle("CV Screening Multi-Agent System Report")
    c.setFont("Helvetica", 11)

    for raw in lines:
        if raw.startswith("# "):
            c.setFont("Helvetica-Bold", 15)
            wrapped = _wrap_text(raw[2:], 74)
        elif raw.startswith("## "):
            c.setFont("Helvetica-Bold", 13)
            wrapped = _wrap_text(raw[3:], 82)
        elif raw.startswith("### "):
            c.setFont("Helvetica-Bold", 12)
            wrapped = _wrap_text(raw[4:], 86)
        else:
            c.setFont("Helvetica", 11)
            wrapped = _wrap_text(raw, 92)

        if not wrapped:
            wrapped = [""]

        for line in wrapped:
            if y <= 48:
                c.showPage()
                c.setFont("Helvetica", 11)
                y = height - top_margin
            c.drawString(left_margin, y, line)
            y -= line_height

    c.save()
    print(f"PDF exported to: {out_pdf}")


if __name__ == "__main__":
    main()
