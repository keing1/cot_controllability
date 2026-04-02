"""Render side-by-side Base vs FT transcript comparison PNGs.

Reads a JSON file with transcript pairs and renders each as a two-column
PNG: left column = base model, right column = fine-tuned model.

Usage:
    # 1. First generate the data file (see prepare_compare_data.py)
    # 2. Then render:
    python scripts/analysis/render_transcript_compare.py \
        --data /tmp/compare_data.json

JSON format (list of objects):
    [{
        "label": "GPT-OSS-120B",
        "outpath": "results/summaries/transcripts/compare_foo.png",
        "base_user_prompt": "...",
        "base_reasoning": "...",
        "base_response": "...",
        "ft_reasoning": "...",
        "ft_response": "..."
    }, ...]
"""
import argparse
import json
import textwrap
from PIL import Image, ImageDraw, ImageFont

# ── Layout constants ──
TOTAL_WIDTH = 880
PAD = 16
THINK_PAD_TOP = 10
THINK_PAD_BOT = 10
BOX_PAD_BOT = 10
GAP = 12
FONT_SIZE = 14
LABEL_SIZE = 16
LINE_SPACING = 4
CORNER = 12

FONT = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", FONT_SIZE)
LABEL_FONT = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", LABEL_SIZE, index=1)

# ── Colors ──
USER_BG, USER_BD = "#EBEBEB", "#C0C0C0"
ASST_BG, ASST_BD = "#D9F2D9", "#8CC98C"
THINK_BG, THINK_BD = "#C0E8C0", "#70B870"
LABEL_CLR = "#1A1A1A"
TEXT_CLR = "#1A1A1A"
FT_RED = "#CC0000"

# ── Derived metrics ──
_sample = "A" * 50
CHAR_W = FONT.getlength(_sample) / 50

COL_W = (TOTAL_WIDTH - GAP) // 2
COL_INNER = COL_W - 2 * PAD
COL_WRAP = int(COL_INNER / CHAR_W) - 2
THINK_INNER = COL_INNER - 2 * PAD
THINK_WRAP = int(THINK_INNER / CHAR_W) - 2
USER_INNER = TOTAL_WIDTH - 2 * PAD
USER_WRAP = int(USER_INNER / CHAR_W) - 2

CHAR_SUBS = {'\u210F': '\u0127'}

_measure_img = Image.new("RGB", (1, 1))
_measure_draw = ImageDraw.Draw(_measure_img)


def fix_chars(text):
    for old, new in CHAR_SUBS.items():
        text = text.replace(old, new)
    return text


def wrap(text, width):
    text = fix_chars(text)
    lines = []
    for raw in text.split("\n"):
        if raw.strip() == "":
            lines.append("")
        else:
            lines.extend(textwrap.wrap(raw, width=width, break_long_words=True, break_on_hyphens=False) or [""])
    return "\n".join(lines)


def measure_text_height(text, font):
    bbox = _measure_draw.multiline_textbbox((0, 0), text, font=font, spacing=LINE_SPACING)
    return bbox[3]


def draw_rounded_rect(draw, xy, fill, outline, radius):
    draw.rounded_rectangle(xy, radius=radius, fill=fill, outline=outline, width=2)


def make_think_text(raw_think, wrap_width):
    if raw_think:
        inner = wrap(raw_think, wrap_width)
        return f"<think>\n{inner}\n</think>"
    else:
        return "<think>\n</think>"


def col_height(think_text, resp_text):
    h = PAD + LABEL_SIZE + 8
    if think_text:
        th = measure_text_height(think_text, FONT)
        h += THINK_PAD_TOP + th + THINK_PAD_BOT + 8
    if resp_text:
        rh = measure_text_height(resp_text, FONT)
        h += rh
    h += BOX_PAD_BOT
    return h


def draw_col(draw, x, y, w, h, label, think_text, resp_text, is_ft=False):
    draw_rounded_rect(draw, (x, y, x + w - 1, y + h - 1), ASST_BG, ASST_BD, CORNER)
    lx = x + PAD
    ly = y + PAD
    if is_ft:
        draw.text((lx, ly), "Assistant ", font=LABEL_FONT, fill=LABEL_CLR)
        pw = LABEL_FONT.getlength("Assistant ")
        draw.text((lx + pw, ly), "(FT)", font=LABEL_FONT, fill=FT_RED)
    else:
        draw.text((lx, ly), "Assistant ", font=LABEL_FONT, fill=LABEL_CLR)
        pw = LABEL_FONT.getlength("Assistant ")
        draw.text((lx + pw, ly), f"({label})", font=LABEL_FONT, fill=FT_RED)

    cy = ly + LABEL_SIZE + 8
    if think_text:
        th = measure_text_height(think_text, FONT)
        think_box_h = THINK_PAD_TOP + th + THINK_PAD_BOT
        draw_rounded_rect(draw, (x + PAD, cy, x + w - PAD - 1, cy + think_box_h - 1),
                          THINK_BG, THINK_BD, CORNER // 2)
        draw.multiline_text((x + 2 * PAD, cy + THINK_PAD_TOP), think_text, font=FONT, fill=TEXT_CLR, spacing=LINE_SPACING)
        cy += think_box_h + 8
    if resp_text:
        draw.multiline_text((x + PAD, cy), resp_text, font=FONT, fill=TEXT_CLR, spacing=LINE_SPACING)


def render_comparison(item):
    user_text = wrap(item["base_user_prompt"], USER_WRAP)
    base_think = make_think_text(item["base_reasoning"], THINK_WRAP)
    base_resp = wrap(item["base_response"], COL_WRAP)
    ft_think = make_think_text(item["ft_reasoning"], THINK_WRAP)
    ft_resp = wrap(item["ft_response"], COL_WRAP)

    user_h = measure_text_height(user_text, FONT)
    user_box_h = PAD + LABEL_SIZE + 8 + user_h + PAD

    base_h = col_height(base_think, base_resp)
    ft_h = col_height(ft_think, ft_resp)
    col_h = max(base_h, ft_h)
    total_h = user_box_h + GAP + col_h

    img = Image.new("RGBA", (TOTAL_WIDTH, total_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # User box (full width)
    draw_rounded_rect(draw, (0, 0, TOTAL_WIDTH - 1, user_box_h - 1), USER_BG, USER_BD, CORNER)
    draw.text((PAD, PAD), "User", font=LABEL_FONT, fill=LABEL_CLR)
    draw.multiline_text((PAD, PAD + LABEL_SIZE + 8), user_text, font=FONT, fill=TEXT_CLR, spacing=LINE_SPACING)

    # Two assistant columns
    y_col = user_box_h + GAP
    draw_col(draw, 0, y_col, COL_W, col_h, item["label"], base_think, base_resp)
    draw_col(draw, COL_W + GAP, y_col, COL_W, col_h, item["label"], ft_think, ft_resp, is_ft=True)

    img.save(item["outpath"])
    print(f"Saved {item['outpath']}  size={img.size}")


def main():
    parser = argparse.ArgumentParser(description='Render base-vs-FT transcript comparison PNGs')
    parser.add_argument('--data', required=True, help='Path to JSON file with transcript pairs')
    args = parser.parse_args()

    with open(args.data) as f:
        items = json.load(f)

    print(f"TOTAL_WIDTH={TOTAL_WIDTH}  COL_W={COL_W}  COL_WRAP={COL_WRAP}  "
          f"THINK_WRAP={THINK_WRAP}  USER_WRAP={USER_WRAP}")

    for item in items:
        render_comparison(item)


if __name__ == '__main__':
    main()
