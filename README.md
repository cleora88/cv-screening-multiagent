# CV Screening Multi-Agent System

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cleora88/cv-screening-multiagent/blob/main/CV_Screening_MultiAgent_Colab_Kaggle.ipynb)

Multi-agent CV screening project for ranking candidates against job descriptions. The system combines two specialist agents, an orchestrator, a trained PyTorch model, skill-gap analysis, human review checkpoints, and JSON audit logs.

## What It Does

- Screens one CV or a batch of CVs against a selected job.
- Ranks candidates as `shortlist`, `review`, or `reject`.
- Shows technical fit, profile fit, matched skills, missing skills, and decision rationale.
- Flags uncertain cases for human review.
- Supports optional Ollama/CrewAI execution for local demos.

## Architecture

- **Technical Matcher**: uses the PyTorch model and skill extractor.
- **Profile Analyzer**: evaluates experience, education, projects, teamwork, and communication signals.
- **Orchestrator**: combines specialist outputs, applies thresholds, and triggers HITL review.

Tools:

- `models/cv_fit_model.pt`: trained PyTorch fit classifier.
- `src/tools/skill_extractor_tool.py`: matched/missing skill analysis.

## Quick Start

Recommended local runtime: Python 3.11 or 3.12.

```powershell
python -m pip install -r requirements.txt
python -m src.main --runtime deterministic
```

Run the Streamlit frontend:

```powershell
python -m streamlit run src/app_frontend.py
```

Windows demo launcher with Ollama checks:

```powershell
.\run_project.ps1 -SkipDependencyInstall -Port 8501
```

## Colab / Kaggle

The notebook can run by itself in Colab. If project files are missing, it clones this repository automatically.

[Open the notebook in Colab](https://colab.research.google.com/github/cleora88/cv-screening-multiagent/blob/main/CV_Screening_MultiAgent_Colab_Kaggle.ipynb)

## Useful Commands

Train or regenerate the model:

```powershell
python -m src.train_baseline
```

Run batch screening:

```powershell
python -m src.main --runtime deterministic --batch-screening --cv-file data/sample_cvs.json --job-file data/sample_jobs.json
```

Run evaluation:

```powershell
python -m src.evaluate_pipeline
```

Run tests:

```powershell
python -m pytest -q
```

Check full local demo readiness:

```powershell
python -m src.preflight_demo
```

## Project Evidence

- Trained model: `models/cv_fit_model.pt`
- Model metrics: `models/model_evaluation.md`
- Pipeline evaluation: `logs/pipeline_evaluation.json`
- Report PDF: `submission/final_report.pdf`
- Slides: `submission/presentation_slides.md`
- Demo video link: `DEMO_VIDEO.md`

## Requirement Coverage

- 2 specialist agents + 1 orchestrator
- PyTorch model trained and used as an agent tool
- At least 2 tools with clear inputs/outputs
- Human-in-the-loop checkpoint
- Error handling and fallback behavior
- JSON logging for agent actions
- Tests, pipeline evaluation, report, slides, and Colab notebook

## Notes

CrewAI and LangGraph are intended for Python 3.11/3.12. The deterministic runtime is kept as a reliable fallback for hosted notebooks and environments where optional agent dependencies are unavailable.
