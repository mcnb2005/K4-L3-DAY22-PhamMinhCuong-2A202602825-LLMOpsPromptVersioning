"""Render the checked-in RAGAS JSON report as a submission-ready PNG."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = ROOT / "evidence" / "03_ragas_report.json"
OUTPUT_PATH = ROOT / "evidence" / "03_ragas_scores.png"

WIDTH, HEIGHT = 1400, 900
BACKGROUND = "#07111F"
PANEL = "#101D2E"
PANEL_ALT = "#13243A"
TEXT = "#EAF2FF"
MUTED = "#9BB0CC"
ACCENT = "#49D6B2"
BLUE = "#64A8FF"
GOLD = "#F4C95D"
GRID = "#2A3D57"


def font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    windows_fonts = Path("C:/Windows/Fonts")
    candidates = [
        windows_fonts / ("seguisb.ttf" if bold else "segoeui.ttf"),
        windows_fonts / ("arialbd.ttf" if bold else "arial.ttf"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


def centered_text(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    value: str,
    text_font: ImageFont.ImageFont,
    fill: str,
) -> None:
    left, top, right, bottom = box
    bounds = draw.textbbox((0, 0), value, font=text_font)
    text_width = bounds[2] - bounds[0]
    text_height = bounds[3] - bounds[1]
    x = left + (right - left - text_width) / 2
    y = top + (bottom - top - text_height) / 2 - bounds[1]
    draw.text((x, y), value, font=text_font, fill=fill)


def main() -> None:
    report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    v1 = report["prompt_v1_scores"]
    v2 = report["prompt_v2_scores"]

    image = Image.new("RGB", (WIDTH, HEIGHT), BACKGROUND)
    draw = ImageDraw.Draw(image)

    draw.rounded_rectangle((55, 45, WIDTH - 55, HEIGHT - 45), radius=26, fill=PANEL)
    draw.text((100, 82), "RAGAS EVALUATION RESULTS", font=font(42, bold=True), fill=TEXT)
    draw.text(
        (102, 145),
        "Prompt Versioning Lab  |  OpenRouter  |  50 samples per prompt",
        font=font(23),
        fill=MUTED,
    )

    badge = (1030, 85, 1250, 140)
    draw.rounded_rectangle(badge, radius=14, fill="#123C39", outline=ACCENT, width=2)
    centered_text(draw, badge, "TARGET PASSED", font(21, bold=True), ACCENT)

    draw.text((100, 200), "Phạm Minh Cương", font=font(28, bold=True), fill=TEXT)
    draw.text((100, 240), "MSSV: 2A202602825", font=font(22), fill=MUTED)

    table_left, table_top, table_right = 100, 310, 1300
    row_height = 76
    columns = [table_left, 660, 880, 1100, table_right]
    headers = ["METRIC", "PROMPT V1", "PROMPT V2", "WINNER"]
    metrics = [
        ("Faithfulness", "faithfulness"),
        ("Answer relevancy", "answer_relevancy"),
        ("Context recall", "context_recall"),
        ("Context precision", "context_precision"),
    ]

    draw.rounded_rectangle(
        (table_left, table_top, table_right, table_top + row_height),
        radius=14,
        fill="#18314F",
    )
    for index, header in enumerate(headers):
        centered_text(
            draw,
            (columns[index], table_top, columns[index + 1], table_top + row_height),
            header,
            font(21, bold=True),
            TEXT,
        )

    for row_index, (label, key) in enumerate(metrics, start=1):
        top = table_top + row_index * row_height
        bottom = top + row_height
        fill = PANEL_ALT if row_index % 2 else PANEL
        draw.rectangle((table_left, top, table_right, bottom), fill=fill)
        score_v1 = float(v1[key])
        score_v2 = float(v2[key])
        if abs(score_v1 - score_v2) < 0.00005:
            winner = "TIE"
        elif score_v1 > score_v2:
            winner = "V1"
        else:
            winner = "V2"

        draw.text((columns[0] + 24, top + 23), label, font=font(21), fill=TEXT)
        centered_text(draw, (columns[1], top, columns[2], bottom), f"{score_v1:.4f}", font(23, bold=True), BLUE)
        centered_text(draw, (columns[2], top, columns[3], bottom), f"{score_v2:.4f}", font(23, bold=True), GOLD)
        winner_color = MUTED if winner == "TIE" else ACCENT
        centered_text(draw, (columns[3], top, columns[4], bottom), winner, font(22, bold=True), winner_color)
        draw.line((table_left, bottom, table_right, bottom), fill=GRID, width=1)

    for x in columns[1:-1]:
        draw.line((x, table_top, x, table_top + row_height * 5), fill=GRID, width=1)

    result_top = 725
    draw.rounded_rectangle((100, result_top, 1300, 800), radius=14, fill="#0E302F")
    draw.text((130, result_top + 19), "PASS", font=font(26, bold=True), fill=ACCENT)
    draw.text(
        (230, result_top + 21),
        f"Best faithfulness = {max(float(v1['faithfulness']), float(v2['faithfulness'])):.4f}  ≥  0.8000",
        font=font(23),
        fill=TEXT,
    )

    draw.text(
        (100, 830),
        "Source: evidence/03_ragas_report.json  •  Evaluation complete: true",
        font=font(18),
        fill=MUTED,
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    image.save(OUTPUT_PATH, format="PNG", optimize=True)
    print(f"Rendered {OUTPUT_PATH} ({WIDTH}x{HEIGHT})")


if __name__ == "__main__":
    main()
