# CV Screening Multi-Agent System - Defense Slides

## 1. Problem and Motivation

- Manual CV screening is slow and inconsistent.
- Goal: improve consistency and traceability with specialist agents and human governance.

## 2. Why Multi-Agent

- Technical Matcher focuses on hard-skill fit.
- Profile Analyzer focuses on seniority and role-fit signals.
- Orchestrator combines evidence and enforces policy.

## 3. Architecture

- Inputs: CV text + Job text.
- Tool 1: PyTorch model tool for fit signal.
- Tool 2: Skill extractor for coverage and gaps.
- Orchestrator computes final score and recommendation.

## 4. Deep Learning Model

- Framework: PyTorch 2.x.
- Artifact: models/cv_fit_model.pt.
- Evaluation: accuracy, confusion matrix, precision/recall/F1.

## 5. Workflow (End-to-End)

1. Ingest CV and job.
2. Technical and profile agents produce outputs.
3. Orchestrator applies weighted decision logic.
4. HITL checkpoint triggered for borderline/conflict/failure cases.
5. Final output + JSONL logs.

## 6. Human-in-the-Loop

- Strict mode requires explicit reviewer decision.
- Streamlit single-screening blocks until reviewer confirms Shortlist/Reject/Needs Review.

## 7. Guardrails and Error Handling

- Input validation for file existence, format, required fields.
- Tool-level fallbacks for model and extraction failures.
- Ollama errors handled with non-crashing fallback message.

## 8. Logging and Traceability

- Structured JSONL with timestamp, agent_name, action, tool_used, status, error.
- Event payload preserves detailed context for audit and debugging.

## 9. Evaluation and Robustness

- Automated tests for smoke + edge cases.
- Pipeline evaluation on labeled cases with accuracy and confusion matrix.

## 10. Limitations and Next Steps

- Expand dataset realism and scale.
- Add semantic skill matching and fairness diagnostics.
- Extend strict HITL to batch-level review workflows.
