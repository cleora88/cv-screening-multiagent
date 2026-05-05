# CV Screening Multi-Agent System

Multi-agent AI project for the UIR Integrated Project brief. The domain is CV screening: specialized agents collaborate to assess how well a candidate CV matches a job description, while a human reviewer remains in control for borderline or conflicted cases.

## What this project demonstrates

- 2 specialist agents and 1 orchestrator
- PyTorch model integrated as a real tool in the screening loop
- 2 tools with clear inputs and outputs
- Human-in-the-loop checkpoint for borderline or conflicting decisions
- JSON logging for traceability
- Batch evaluation and edge-case tests for robustness

## Agent design

1. Technical Matcher
   - Uses the DL model tool and skill extractor.
   - Produces hard-skill fit, matched skills, and missing requirements.

2. Profile Analyzer
   - Scores seniority, years of experience, project ownership, communication, and education signals.

3. Orchestrator
   - Combines both specialist assessments.
   - Detects disagreement and triggers human review when needed.

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
```

Run the optional CrewAI orchestration path:

```bash
python -m src.main --use-crewai
```

Run the pipeline on custom JSON input files:

```bash
python -m src.main --cv-file data/sample_cvs.json --job-file data/sample_jobs.json
```

Run batch evaluation on labeled CV screening cases:

```bash
python -m src.evaluate_pipeline
```

Run the web frontend (Streamlit):

```bash
streamlit run src/app_frontend.py
```

Recommended demo launcher (Windows PowerShell) that ensures dependencies, starts Ollama, pulls the model if missing, then opens Streamlit:

```powershell
./run_project.ps1
```

Optional flags:

```powershell
./run_project.ps1 -Port 8502 -OllamaModel llama3.2 -SkipDependencyInstall
```

The frontend includes a "Download Run Report (JSON)" button to export each screening result for demo or submission evidence.

Run tests:

```bash
pytest -q
```

## Output artifacts

- models/cv_fit_model.pt: trained PyTorch weights
- models/model_evaluation.json: held-out test metrics
- models/model_evaluation.md: readable evaluation summary
- logs/run_*.jsonl: per-action runtime logs
- logs/pipeline_evaluation.json: end-to-end pipeline evaluation summary
- src/app_frontend.py: interactive demo UI for live presentation

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
      pipeline/           # Human-in-the-loop checkpoint logic
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
- Working multi-agent system: CrewAI-first runtime with deterministic fallback
- Evaluation and robustness: tests plus batch pipeline evaluation
- Reproducibility: setup and run commands documented here

## CrewAI Primary + LangGraph Stretch Goal

- CrewAI is treated as the primary runtime backend through `--runtime auto` and `--runtime crewai`.
- Deterministic runtime remains as resilience fallback when CrewAI is unavailable.
- LangGraph scaffold is included in `src/langgraph_runtime.py` as a stretch-goal foundation for graph-based orchestration.

## Notes for the defense

- The model tool uses the trained PyTorch model when weights are available.
- If the model file is missing or incompatible, the system falls back to a deterministic heuristic instead of random behavior.
- Human review is triggered for borderline scores, specialist disagreement, or agent failures.
