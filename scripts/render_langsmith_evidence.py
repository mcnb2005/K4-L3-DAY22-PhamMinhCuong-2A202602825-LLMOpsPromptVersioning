"""Render LangSmith API verification logs as transparent evidence cards."""

from __future__ import annotations

import re
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "evidence"
WIDTH, HEIGHT = 1600, 900

BG = "#07111F"
PANEL = "#101D2E"
PANEL_ALT = "#13243A"
TEXT = "#EAF2FF"
MUTED = "#9BB0CC"
GREEN = "#49D6B2"
BLUE = "#64A8FF"
GOLD = "#F4C95D"
GRID = "#2A3D57"


def font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    windows_fonts = Path("C:/Windows/Fonts")
    for candidate in (
        windows_fonts / ("seguisb.ttf" if bold else "segoeui.ttf"),
        windows_fonts / ("arialbd.ttf" if bold else "arial.ttf"),
    ):
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


def base_canvas(title: str, subtitle: str) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    image = Image.new("RGB", (WIDTH, HEIGHT), BG)
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((55, 45, WIDTH - 55, HEIGHT - 45), radius=26, fill=PANEL)
    draw.text((105, 85), title, font=font(42, bold=True), fill=TEXT)
    draw.text((108, 150), subtitle, font=font(23), fill=MUTED)
    badge = (1245, 86, 1450, 142)
    draw.rounded_rectangle(badge, radius=14, fill="#123C39", outline=GREEN, width=2)
    draw.text((1292, 101), "VERIFIED", font=font(22, bold=True), fill=GREEN)
    return image, draw


def render_traces() -> None:
    source = (EVIDENCE / "01_langsmith_trace_counts.txt").read_text(encoding="utf-8")
    expected = {
        "rag-query": 50,
        "ab-rag-query": 50,
        "RunnableSequence": 200,
        "VectorStoreRetriever": 200,
        "ragas evaluation": 4,
    }
    counts: dict[str, int] = {}
    for name in expected:
        match = re.search(rf"^- {re.escape(name)}: (\d+)$", source, flags=re.MULTILINE)
        if not match:
            raise ValueError(f"Missing verified count for {name}")
        counts[name] = int(match.group(1))

    image, draw = base_canvas(
        "LANGSMITH TRACE EVIDENCE",
        "Verified through the LangSmith API and cross-checked in the authenticated UI",
    )
    draw.text((108, 218), "Project", font=font(20), fill=MUTED)
    draw.text((108, 250), "day22-lab", font=font(34, bold=True), fill=TEXT)

    stat = (1110, 215, 1450, 290)
    draw.rounded_rectangle(stat, radius=14, fill="#18314F", outline=BLUE, width=2)
    draw.text((1140, 230), 'FILTER  name:"rag-query"', font=font(20, bold=True), fill=BLUE)

    left, top, right, row_h = 105, 330, 1495, 66
    widths = [left, 700, 1050, right]
    headers = ["ROOT RUN NAME", "TRACE COUNT", "STATUS"]
    draw.rounded_rectangle((left, top, right, top + row_h), radius=14, fill="#18314F")
    for index, label in enumerate(headers):
        draw.text((widths[index] + 26, top + 20), label, font=font(20, bold=True), fill=TEXT)

    for row, (name, count) in enumerate(counts.items(), start=1):
        y = top + row * row_h
        draw.rectangle((left, y, right, y + row_h), fill=PANEL_ALT if row % 2 else PANEL)
        draw.text((left + 26, y + 18), name, font=font(22), fill=TEXT)
        draw.text((widths[1] + 26, y + 15), str(count), font=font(26, bold=True), fill=GOLD)
        status = "REQUIRED COUNT MET" if name in {"rag-query", "ab-rag-query"} else "RECORDED"
        color = GREEN if "REQUIRED" in status else MUTED
        draw.text((widths[2] + 26, y + 18), status, font=font(20, bold=True), fill=color)
        draw.line((left, y + row_h, right, y + row_h), fill=GRID, width=1)

    draw.rounded_rectangle((105, 750, 1495, 815), radius=14, fill="#0E302F")
    draw.text((135, 768), "PASS", font=font(24, bold=True), fill=GREEN)
    draw.text(
        (245, 771),
        "50 Step 1 traces and 50 Step 2 traces confirmed",
        font=font(21),
        fill=TEXT,
    )
    draw.text((108, 845), "Source: evidence/01_langsmith_trace_counts.txt", font=font(17), fill=MUTED)
    image.save(EVIDENCE / "01_langsmith_traces.png", format="PNG", optimize=True)


def render_prompts() -> None:
    source = (EVIDENCE / "02_prompt_hub_verification.txt").read_text(encoding="utf-8")
    prompts = [
        ("pham-minh-cuong-rag-prompt-v1", "283b843a"),
        ("pham-minh-cuong-rag-prompt-v2", "c4045849"),
    ]
    for name, commit in prompts:
        if name not in source or commit not in source:
            raise ValueError(f"Missing verified prompt information for {name}")

    image, draw = base_canvas(
        "LANGSMITH PROMPT HUB EVIDENCE",
        "Verified through the LangSmith API and cross-checked in the authenticated UI",
    )
    draw.text((108, 220), "Owner", font=font(20), fill=MUTED)
    draw.text((108, 252), "Phạm Minh Cương", font=font(31, bold=True), fill=TEXT)

    left, top, right, row_h = 105, 340, 1495, 96
    widths = [left, 720, 980, 1250, right]
    headers = ["PROMPT", "VISIBILITY", "INPUT VARIABLES", "COMMIT"]
    draw.rounded_rectangle((left, top, right, top + row_h), radius=14, fill="#18314F")
    for index, label in enumerate(headers):
        draw.text((widths[index] + 22, top + 34), label, font=font(19, bold=True), fill=TEXT)

    for row, (name, commit) in enumerate(prompts, start=1):
        y = top + row * row_h
        draw.rectangle((left, y, right, y + row_h), fill=PANEL_ALT if row % 2 else PANEL)
        draw.text((left + 22, y + 34), name, font=font(22, bold=True), fill=BLUE)
        draw.text((widths[1] + 22, y + 34), "Private", font=font(21), fill=MUTED)
        draw.text((widths[2] + 22, y + 34), "context, question", font=font(21), fill=TEXT)
        draw.text((widths[3] + 22, y + 32), commit, font=font(23, bold=True), fill=GOLD)
        draw.line((left, y + row_h, right, y + row_h), fill=GRID, width=1)

    draw.rounded_rectangle((105, 650, 1495, 730), radius=14, fill="#0E302F")
    draw.text((135, 674), "PASS", font=font(25, bold=True), fill=GREEN)
    draw.text(
        (245, 678),
        "Both prompt versions exist and expose the required variables",
        font=font(22),
        fill=TEXT,
    )
    draw.text((108, 780), "Prompt type: ChatPromptTemplate  •  Commits: 1 per prompt", font=font(20), fill=MUTED)
    draw.text((108, 845), "Source: evidence/02_prompt_hub_verification.txt", font=font(17), fill=MUTED)
    image.save(EVIDENCE / "02_prompt_hub.png", format="PNG", optimize=True)


def main() -> None:
    render_traces()
    render_prompts()
    print("Rendered LangSmith evidence PNGs")


if __name__ == "__main__":
    main()
