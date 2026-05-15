"""Brutalist Wealth Report PDF."""

from datetime import date
from io import BytesIO

from reportlab.lib import colors
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from src.tools import forecast


def _money(v: float) -> str:
    # Built-in Helvetica is a Type 1 font with WinAnsi encoding that doesn't
    # have U+20B1 (₱). Use the ISO 4217 code instead so the PDF actually
    # renders without registering a TrueType font.
    return f"PHP {v:,.0f}"


def build_report(state: dict) -> bytes:
    buf = BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=LETTER, leftMargin=0.6 * inch,
                              rightMargin=0.6 * inch, topMargin=0.6 * inch,
                              bottomMargin=0.6 * inch)
    styles = getSampleStyleSheet()
    title = ParagraphStyle("title", parent=styles["Title"], fontName="Helvetica-Bold",
                            fontSize=24, leading=28, textColor=colors.black)
    h2 = ParagraphStyle("h2", parent=styles["Heading2"], fontName="Helvetica-Bold",
                         fontSize=12, leading=14, spaceBefore=14, spaceAfter=4,
                         textColor=colors.black)
    body = ParagraphStyle("body", parent=styles["BodyText"], fontName="Helvetica",
                           fontSize=10, leading=12)

    story = []
    story.append(Paragraph("THE WAD — WEALTH REPORT", title))
    story.append(Paragraph(date.today().isoformat(), body))
    story.append(Spacer(1, 12))

    assets = state.get("assets", [])
    nw = forecast.total_net_worth(assets)
    cb = forecast.total_cost_basis(assets)
    cf = forecast.cashflow(state.get("income", []), state.get("fixed_outflow", []),
                              state.get("variable_burn", []))
    yr = forecast.annual_yield(assets, state.get("settings", {}))
    profile = state.get("risk_profile") or "—"

    summary = [
        ["Net Worth", _money(nw)],
        ["Cost Basis", _money(cb)],
        ["Lifetime ROI", f"{((nw - cb) / cb * 100):.1f}%" if cb else "—"],
        ["Monthly Inflow", _money(cf.inflow)],
        ["Monthly Burn", _money(cf.burn)],
        ["Monthly Surplus", _money(cf.surplus)],
        ["Est. Annual Yield", _money(yr)],
        ["Hourly Yield (8,760h)", f"PHP {yr/8760:.2f}"],
        ["Risk Profile", profile],
    ]
    t = Table(summary, colWidths=[2.5 * inch, 3.0 * inch])
    t.setStyle(TableStyle([
        ("FONT", (0, 0), (-1, -1), "Helvetica", 10),
        ("FONT", (0, 0), (0, -1), "Helvetica-Bold", 10),
        ("LINEBELOW", (0, 0), (-1, -1), 0.25, colors.black),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(t)

    story.append(Paragraph("ASSET LEDGER", h2))
    rows = [["Name", "Type", "Cost Basis", "Current Value", "ROI"]]
    for a in assets:
        rows.append([
            a.get("name", ""), a.get("type", ""),
            _money(float(a.get("purchase_price", 0) or 0)),
            _money(float(a.get("current_value", 0) or 0)),
            f"{forecast.asset_roi(a) * 100:.1f}%",
        ])
    if len(rows) == 1:
        rows.append(["—", "—", "—", "—", "—"])
    at = Table(rows, colWidths=[1.7 * inch, 1.1 * inch, 1.2 * inch, 1.3 * inch, 0.9 * inch])
    at.setStyle(TableStyle([
        ("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 10),
        ("FONT", (0, 1), (-1, -1), "Helvetica", 9),
        ("LINEBELOW", (0, 0), (-1, 0), 1, colors.black),
        ("LINEBELOW", (0, 1), (-1, -1), 0.25, colors.grey),
    ]))
    story.append(at)

    story.append(Paragraph("FORECAST", h2))
    er = float(state.get("settings", {}).get("expected_return", 0.07))
    proj_10 = forecast.project_net_worth(nw, max(0.0, cf.surplus), er, 10)[-1]
    proj_20 = forecast.project_net_worth(nw, max(0.0, cf.surplus), er, 20)[-1]
    proj_30 = forecast.project_net_worth(nw, max(0.0, cf.surplus), er, 30)[-1]
    fwd = [["10 Years", _money(proj_10)], ["20 Years", _money(proj_20)],
            ["30 Years", _money(proj_30)]]
    ft = Table(fwd, colWidths=[2.5 * inch, 3.0 * inch])
    ft.setStyle(TableStyle([
        ("FONT", (0, 0), (-1, -1), "Helvetica", 10),
        ("FONT", (0, 0), (0, -1), "Helvetica-Bold", 10),
        ("LINEBELOW", (0, 0), (-1, -1), 0.25, colors.black),
    ]))
    story.append(ft)

    fd = forecast.freedom_date(nw, cf.burn, max(0.0, cf.surplus), er)
    story.append(Paragraph("FREEDOM DATE", h2))
    if fd["reached"]:
        story.append(Paragraph(f"Estimated date: <b>{fd['date']}</b> (in {fd['years']:.1f} years).", body))
        story.append(Paragraph(f"Required net worth at blended return: {_money(fd['needed_nw'])}.", body))
    else:
        story.append(Paragraph("Not reachable within projection horizon at current trajectory.", body))

    doc.build(story)
    return buf.getvalue()
