# CV Screening Multi-Agent Architecture

## Problem choice

This project implements an open-domain multi-agent AI system for CV screening. The goal is to help recruiters rank candidates against a target job description while keeping a human approval checkpoint for borderline or conflicted decisions.

## Agent roles

1. Technical Matcher
   - Uses the PyTorch model tool and the skill extraction tool.
   - Evaluates hard-skill coverage, missing required skills, and model-based fit score.

2. Profile Analyzer
   - Focuses on experience level, seniority, project ownership, communication signals, and education markers.

3. Orchestrator
   - Combines specialist outputs.
   - Detects disagreement and borderline cases.
   - Triggers human review when confidence is not strong enough.

## Tools

1. DL Model Tool
   - Input: CV text and job text.
   - Output: fit label, score, feature vector, matched skills, missing skills, model source.
   - Backed by a PyTorch classifier trained on synthetic CV/job examples.

2. Skill Extractor Tool
   - Input: CV text and job text.
   - Output: matched skills, missing required skills, job skill list, coverage ratio.

## Workflow

1. Load CV and job description.
2. Technical Matcher runs the model tool and skill extractor.
3. Profile Analyzer scores experience and role-fit evidence.
4. Orchestrator computes a weighted final score.
5. If the score is borderline, an agent fails, or the specialists disagree, a human checkpoint is required.
6. Final JSON result and JSONL logs are written to disk.

## Brief alignment

- 2 specialists + 1 orchestrator: satisfied
- PyTorch model trained by the team: satisfied via src/train_baseline.py
- 2 tools with defined inputs/outputs: satisfied
- Human-in-the-loop checkpoint: satisfied
- Error handling and no-crash fallback: satisfied
- JSON logging of actions: satisfied
- Reproducible README and evaluation commands: satisfied

## Demo path

1. Train the model.
2. Run one candidate through the pipeline.
3. Run batch evaluation on labeled cases.
4. Show JSON logs and explain a human-review example.
