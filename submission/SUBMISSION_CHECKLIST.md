# Submission Checklist

- [x] Source code repository
- [x] Trained model artifact (`models/cv_fit_model.pt`)
- [x] Structured runtime logs (`logs/run_*.jsonl`)
- [x] Pipeline evaluation artifact (`logs/pipeline_evaluation.json`)
- [x] Report source (`REPORT.md`)
- [x] Slides source (`submission/presentation_slides.md`)
- [ ] Demo video (3-5 minutes)

## Report Export

Generate PDF from the report source:

```powershell
python scripts/export_report_pdf.py
```

Expected output: `submission/final_report.pdf`
