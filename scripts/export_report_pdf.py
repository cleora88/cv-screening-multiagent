from __future__ import annotations

from pathlib import Path


def _wrap_text(line: str, width: int = 108) -> list[str]:
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
            "reportlab is required to export PDF. Install with: python -m pip install reportlab"
        ) from exc

    lines = report_md.read_text(encoding="utf-8").splitlines()
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
            wrapped = _wrap_text(raw[2:], 82)
        elif raw.startswith("## "):
            c.setFont("Helvetica-Bold", 13)
            wrapped = _wrap_text(raw[3:], 90)
        elif raw.startswith("### "):
            c.setFont("Helvetica-Bold", 12)
            wrapped = _wrap_text(raw[4:], 95)
        else:
            c.setFont("Helvetica", 11)
            wrapped = _wrap_text(raw, 108)

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
