# CV Screening Multi-Agent System

This is our UIR Integrated Project implementation for a simple HR screening use case: shortlisting candidates for a Junior Data Analyst role.

The app lets a recruiter screen one CV, or several CVs at once, against the same job description. Each candidate is analyzed separately, then the system returns a recommendation: `shortlist`, `review`, or `reject`. Borderline or conflicting cases are not treated as final decisions; they are sent to a human review step.

## Main requirements covered

- Two specialist agents plus one orchestrator
- A trained PyTorch model used as a real scoring tool
- A second tool for skill extraction and skill-gap analysis
- Human review for uncertain decisions
- JSON logs for traceability
- Tests and evaluation outputs for robustness

## Agent design

1. Technical Matcher
   - Checks the technical match between the CV and the job.
   - Uses the PyTorch model and the skill extractor.
   - Reports matched skills and skills not found in the CV.

2. Profile Analyzer
   - Looks at experience, education, project work, teamwork, and communication signals.

3. Orchestrator
   - Combines the two agent outputs.
   - Gives the final recommendation.
   - Sends uncertain cases to HR review.

See the architecture note in docs/architecture.md.

## Setup

1. Create and activate a Python 3.10+ environment.
   - Recommended: Python 3.11 or 3.12.
   - CrewAI primary runtime and LangGraph stretch-goal support require Python < 3.14.
   - On Python 3.14, deterministic runtime is used as fallback.
2. Install dependencies.

```bash
pip install -r requirements.txt
```

One-command CrewAI-primary setup on Windows:

```powershell
./setup_crewai_primary.ps1
```

## Colab compatibility

This project is compatible with Colab for training and deterministic pipeline execution.

```python
!python --version
!pip install -r requirements.txt
```

Recommended Colab flow:

1. Upload the repository files or mount Google Drive.
1. Run model training:

```python
!python -m src.train_baseline
```

1. Run deterministic pipeline:

```python
!python -m src.main --runtime deterministic
```

Note: CrewAI runtime is primarily validated in local Python 3.11/3.12 environments.

## Run the project

Train the model and generate evaluation artifacts:

```bash
python -m src.train_baseline
```

Run the deterministic multi-agent pipeline on the sample CV/job pair:

```bash
python -m src.main
```

Runtime selection:

```bash
# CrewAI primary (default behavior in auto mode when available)
python -m src.main --runtime auto

# Force CrewAI
python -m src.main --runtime crewai

# Force deterministic orchestrator
python -m src.main --runtime deterministic

# Compliance mode: require explicit human approval at HITL checkpoints
python -m src.main --runtime deterministic --require-human-approval
```

CrewAI backend policy:

- CrewAI runs enforce an Ollama LLM backend.
- `--runtime crewai` fails fast when Ollama is unreachable.
- `--runtime auto` falls back to deterministic runtime when CrewAI/Ollama is unavailable.

Run the pipeline on custom JSON input files:

```bash
python -m src.main --cv-file data/sample_cvs.json --job-file data/sample_jobs.json
```

Run the simplified HR workflow for one role and one or many CVs:

```bash
python -m src.main --batch-screening --cv-file data/sample_cvs.json --job-file data/sample_jobs.json
```

This screens every CV record against the selected job description, ranks the candidates, keeps the same Technical Matcher + Profile Analyzer + Orchestrator flow for each candidate, and writes JSONL audit logs.

Run batch evaluation on labeled CV screening cases:

```bash
python -m src.evaluate_pipeline
```

Run the web frontend (Streamlit):

```bash
streamlit run src/app_frontend.py
```

Use **Single Screening** for one CV and **Batch Screening** for many CVs against the same Junior Data Analyst job description. In Single Screening mode, enable **Require human approval** to force a blocking reviewer decision before final recommendation output; in Batch Screening, flagged cases are marked for HR review in the ranked table and audit log.

Recommended demo launcher (Windows PowerShell) that ensures dependencies, starts Ollama, pulls the model if missing, then opens Streamlit:

```powershell
./run_project.ps1
```

Optional flags:

```powershell
./run_project.ps1 -Port 8502 -OllamaModel llama3.2 -SkipDependencyInstall
```

The frontend includes JSON/CSV download buttons for single and batch screening results.

Run tests:

```bash
pytest -q
```

Demo preflight (single command PASS/FAIL check for CrewAI + Ollama runtime readiness):

```bash
python -m src.preflight_demo
```

## Output artifacts

- models/cv_fit_model.pt: trained PyTorch weights
- models/model_evaluation.json: held-out test metrics
- models/model_evaluation.md: readable evaluation summary
- logs/run_*.jsonl: per-action runtime logs
- logs/pipeline_evaluation.json: end-to-end pipeline evaluation summary
- src/app_frontend.py: interactive demo UI for live presentation

## Logging schema

Every JSONL event includes:

- timestamp
- agent_name
- action
- tool_used
- input_summary
- output_summary
- status
- error
- event
- payload

This provides consistent traceability for successful and failed actions.

## Project structure

```text
cv-screening-multiagent/
   data/                 # Sample inputs and evaluation datasets
   docs/                 # Architecture and project documentation
   logs/                 # Runtime and evaluation logs (generated)
   models/               # Trained model artifacts (generated)
   src/
      agents/             # Technical agent, profile agent, orchestrator
      data/               # Dataset generation and shared featurization
      pipeline/           # Scoring, batch screening, and HITL checkpoint logic
      tools/              # DL model and skill extraction tools
      utils/              # Shared utilities (JSON logger)
      evaluate_pipeline.py
      main.py
      train_baseline.py
   tests/                # Smoke and edge-case tests
   .editorconfig         # Formatting conventions
   .gitignore            # Ignore generated/local artifacts
   pyproject.toml        # Test configuration metadata
   README.md
   requirements.txt
```

## Alignment with the brief

- System design and architecture: documented in docs/architecture.md
- DL model integration: PyTorch classifier trained via src.train_baseline.py
- Working multi-agent system: CrewAI specialist run + orchestrator decision path with deterministic fallback
- Evaluation and robustness: tests plus batch pipeline evaluation
- Reproducibility: setup and run commands documented here

## Professor Checklist

- Multi-agent architecture (2 specialists + orchestrator): implemented
- CrewAI primary with resilient fallback: implemented
- DL model integrated as a tool in live workflow: implemented
- DL evaluation with confusion matrix and class metrics: implemented
- HITL checkpoint requiring explicit approval: implemented in CLI strict mode and Streamlit single-screening strict mode
- JSONL logging with structured status/action/error fields: implemented
- Edge-case tests and smoke tests: implemented
- Reproducible local and Colab-oriented setup: documented

## Submission checklist

- Source repository: present
- Trained model artifact: `models/cv_fit_model.pt`
- Report source: `REPORT.md`
- Slides source: `submission/presentation_slides.md`
- Report PDF target: `submission/final_report.pdf`

HITL compliance note:

- Use `--require-human-approval` in CLI runs to enforce explicit human approval at checkpoints.
- In strict mode, non-interactive checkpoints are set to `pending-human-approval`.
- In non-strict automated runs, uncertain cases are auto-flagged for review, never auto-approved.

## CrewAI Primary + LangGraph Stretch Goal

- CrewAI is treated as the primary runtime backend through `--runtime auto` and `--runtime crewai`.
- Deterministic runtime remains as resilience fallback when CrewAI is unavailable.
- LangGraph scaffold is included in `src/langgraph_runtime.py` as a stretch-goal foundation for graph-based orchestration.

## Notes for the defense

- The model tool uses the trained PyTorch model when weights are available.
- If the model file is missing or incompatible, the system falls back to a deterministic heuristic instead of random behavior.
- Human review is triggered for borderline scores, specialist disagreement, or agent failures.
