# CV Screening Multi-Agent System - Defense Slides

## 1. Problem and Motivation

- Manual CV screening is slow and inconsistent.
- Use case: Junior Data Analyst hiring shortlist.
- Goal: help HR screen one or many CVs against the same job while keeping human control for uncertain decisions.

## 2. Why Multi-Agent

- Technical Matcher checks hard skills such as Python, SQL, pandas, Excel, and reporting.
- Profile Analyzer checks experience, education, teamwork, communication, and project evidence.
- Orchestrator combines both views and decides shortlist/review/reject.

## 3. Architecture

- Inputs: CV text/PDF + Junior Data Analyst job description.
- Tool 1: PyTorch model tool for fit signal.
- Tool 2: Skill extractor for coverage and gaps.
- Frontend modes: Single Screening and Batch Screening.

## 4. Deep Learning Model

- Framework: PyTorch 2.x.
- Artifact: models/cv_fit_model.pt.
- Evaluation: accuracy, confusion matrix, precision/recall/F1.

## 5. Workflow (End-to-End)

1. HR enters/selects the job description.
2. HR provides one CV or uploads multiple CVs/PDFs.
3. Technical and profile agents produce outputs.
4. Orchestrator applies weighted decision logic.
5. HITL checkpoint triggered for borderline/conflict/failure cases.
6. Final output + JSONL logs.

## 6. Human-in-the-Loop

- Strict mode requires explicit reviewer decision.
- Streamlit single-screening blocks until reviewer confirms Shortlist/Reject/Needs Review.
- Batch mode flags uncertain candidates for HR review in the leaderboard.

## 7. Guardrails and Error Handling

- Input validation for file existence, format, required fields.
- Tool-level fallbacks for model and extraction failures.
- Ollama errors handled with non-crashing fallback message.

## 8. Logging and Traceability

- Structured JSONL with timestamp, agent_name, action, tool_used, status, error.
- Event payload preserves detailed context for audit and debugging.

## 9. Evaluation and Robustness

- 16 automated tests for smoke, edge cases, batch screening, and frontend batch helpers.
- Pipeline evaluation on labeled cases with accuracy and confusion matrix.
- Final pipeline evaluation accuracy: 0.8333.

## 10. Limitations and Next Steps

- Expand dataset realism and scale.
- Add semantic skill matching and fairness diagnostics.
- Add a richer batch-level reviewer workflow.
