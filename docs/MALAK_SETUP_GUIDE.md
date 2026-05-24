# Malak Setup Guide

This guide explains how to run the CV Screening Multi-Agent project on another laptop.

## 1. What The Project Does

The app helps HR screen candidates for one role: **Junior Data Analyst**.

It has two main modes:

- **Single Screening**: analyze one CV against the job description.
- **Batch Screening**: analyze many CVs at once against the same job description.

For each candidate, the system returns:

- final score
- recommendation: `shortlist`, `review`, or `reject`
- technical score
- profile score
- matched skills
- job skills not found in the CV
- audit log evidence

If the system is unsure, it sends the candidate to an HR review step.

## 2. Install Required Software

Install these first:

1. **Python 3.12**
   - Download from: https://www.python.org/downloads/
   - During install, check **Add Python to PATH**.

2. **Git**
   - Download from: https://git-scm.com/downloads

3. **Ollama**
   - Download from: https://ollama.com/download
   - Ollama is used for optional local LLM explanations and CrewAI runtime checks.

## 3. Download The Project

Open PowerShell and run:

```powershell
git clone https://github.com/cleora88/cv-screening-multiagent.git
cd cv-screening-multiagent
```

## 4. Create The Python Environment

Use Python 3.12:

```powershell
py -3.12 -m venv .venv312
.\.venv312\Scripts\activate
```

If activation is blocked, run:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Then activate again:

```powershell
.\.venv312\Scripts\activate
```

## 5. Install Dependencies

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 6. Start Ollama

Open a second PowerShell window and run:

```powershell
ollama serve
```

Then, in another PowerShell window, pull the model:

```powershell
ollama pull llama3.2
```

If Ollama is already running, `ollama serve` may say the port is already in use. That is okay.

## 7. Check Everything Is Ready

Back in the project folder, run:

```powershell
.\.venv312\Scripts\python.exe -m src.preflight_demo
```

Expected result:

```text
PRECHECK PASS
```

## 8. Run The App

The easiest way:

```powershell
.\run_project.ps1
```

Or run Streamlit directly:

```powershell
.\.venv312\Scripts\python.exe -m streamlit run src/app_frontend.py
```

Open the browser at:

```text
http://localhost:8501
```

## 9. How To Use The App

### Single Screening

Use this when you want to test one candidate.

Steps:

1. Choose **Single Screening** in the sidebar.
2. Select sample data, manual input, JSON, or PDF.
3. Click **Run Screening**.
4. Read the result tabs:
   - Intake
   - Agent Analysis
   - Decision Room
   - Evidence
   - Audit Trail
5. If HR review appears, choose:
   - Shortlist
   - Reject
   - Needs Review

### Batch Screening

Use this when you want to compare many candidates.

Steps:

1. Choose **Batch Screening** in the sidebar.
2. Choose CV source:
   - built-in sample CVs
   - JSON list
   - multiple PDF CVs
3. Select the Junior Data Analyst job.
4. Click **Run Batch Screening**.
5. Read the leaderboard.
6. Download JSON or CSV report if needed.

Important: when uploading multiple PDFs, each PDF is treated as a separate CV.

## 10. Useful Commands

Run tests:

```powershell
.\.venv312\Scripts\python.exe -m pytest -q
```

Run one sample candidate in the terminal:

```powershell
.\.venv312\Scripts\python.exe -m src.main --runtime deterministic
```

Run batch screening in the terminal:

```powershell
.\.venv312\Scripts\python.exe -m src.main --batch-screening --cv-file data\sample_cvs.json --job-file data\sample_jobs.json --runtime deterministic
```

Regenerate the final PDF report:

```powershell
.\.venv312\Scripts\python.exe scripts\export_report_pdf.py
```

## 11. Common Problems

### `python` is not recognized

Install Python 3.12 and make sure **Add Python to PATH** is checked.

### PowerShell cannot activate the environment

Run:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### CrewAI is not active

Make sure you are using `.venv312`, not `.venv`.

Check:

```powershell
.\.venv312\Scripts\python.exe --version
```

It should show Python 3.12.

### Ollama is offline

Start it:

```powershell
ollama serve
```

Then pull the model:

```powershell
ollama pull llama3.2
```

### Port 8501 is already used

Run on another port:

```powershell
.\.venv312\Scripts\python.exe -m streamlit run src/app_frontend.py --server.port 8502
```

Then open:

```text
http://localhost:8502
```

## 12. Files To Know

- `src/app_frontend.py`: Streamlit web app
- `src/main.py`: terminal entry point
- `src/agents/`: specialist agents and orchestrator
- `src/tools/`: model tool and skill extractor
- `src/pipeline/`: scoring, HITL, and batch screening logic
- `data/sample_cvs.json`: demo candidates
- `data/sample_jobs.json`: Junior Data Analyst job
- `models/cv_fit_model.pt`: trained PyTorch model
- `logs/pipeline_evaluation.json`: evaluation result
- `submission/CV_Screening_MultiAgent_Report.pdf`: final report PDF
- `submission/CV_Screening_Presentation_EN.pptx`: final presentation
