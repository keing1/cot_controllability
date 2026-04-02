"""Render a single model transcript (User + Assistant) as a PNG.

Usage:
    python scripts/analysis/render_transcript_single.py \
        --user "Question text..." \
        --reasoning "Model thinking..." \
        --response "Model answer..." \
        --outpath results/summaries/transcripts/example.png
"""
import argparse
import textwrap
from PIL import Image, ImageDraw, ImageFont

# ── Layout constants ──
WIDTH = 800
MARGIN = 16
GAP = 12
PADDING = 24
LINE_HEIGHT = 20
WRAP_CHARS = 80
FONT_SIZE = 15
LABEL_FONT_SIZE = 17
RADIUS = 14
BORDER_WIDTH = 2
THINK_PAD = 10
THINK_RADIUS = 8
LABEL_HEIGHT = 34

try:
    font = ImageFont.truetype('/System/Library/Fonts/Menlo.ttc', FONT_SIZE)
    label_font = ImageFont.truetype('/System/Library/Fonts/Helvetica.ttc', LABEL_FONT_SIZE)
except Exception:
    font = ImageFont.load_default()
    label_font = font

# ── Colors ──
USER_BG = '#EBEBEB'
USER_BORDER = '#C0C0C0'
ASST_BG = '#D9F2D9'
ASST_BORDER = '#8CC98C'
THINK_BG = '#C0E8C0'
THINK_BORDER = '#70B870'
LABEL_COLOR = '#1A1A1A'


def wrap_text(text, width=WRAP_CHARS):
    lines = []
    for para in text.split('\n'):
        if para.strip() == '':
            lines.append('')
        else:
            wrapped = textwrap.wrap(para, width=width, break_long_words=True, break_on_hyphens=False)
            lines.extend(wrapped if wrapped else [''])
    return lines


def box_height(lines):
    return LABEL_HEIGHT + PADDING + len(lines) * LINE_HEIGHT + PADDING


def render(user_text, reasoning, response, outpath):
    user_lines = wrap_text(user_text)
    think_lines = wrap_text('<think>\n' + reasoning + '\n</think>')
    response_lines = wrap_text(response)

    think_block_h = len(think_lines) * LINE_HEIGHT + 2 * THINK_PAD
    response_block_h = len(response_lines) * LINE_HEIGHT
    asst_h = LABEL_HEIGHT + PADDING + think_block_h + LINE_HEIGHT + response_block_h + PADDING
    user_h = box_height(user_lines)
    BOX_W = WIDTH - 2 * MARGIN
    total_h = MARGIN + user_h + GAP + asst_h + MARGIN

    img = Image.new('RGBA', (WIDTH, total_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # User box
    ux0, uy0 = MARGIN, MARGIN
    ux1, uy1 = MARGIN + BOX_W, MARGIN + user_h
    draw.rounded_rectangle([ux0, uy0, ux1, uy1], radius=RADIUS, fill=USER_BG, outline=USER_BORDER, width=BORDER_WIDTH)
    draw.text((ux0 + PADDING, uy0 + 8), 'User', fill=LABEL_COLOR, font=label_font)
    y = uy0 + LABEL_HEIGHT + PADDING
    for line in user_lines:
        draw.text((ux0 + PADDING, y), line, fill='#222222', font=font)
        y += LINE_HEIGHT

    # Assistant box
    ax0, ay0 = MARGIN, MARGIN + user_h + GAP
    ax1, ay1 = MARGIN + BOX_W, MARGIN + user_h + GAP + asst_h
    draw.rounded_rectangle([ax0, ay0, ax1, ay1], radius=RADIUS, fill=ASST_BG, outline=ASST_BORDER, width=BORDER_WIDTH)
    draw.text((ax0 + PADDING, ay0 + 8), 'Assistant', fill=LABEL_COLOR, font=label_font)

    # Think block
    ty0 = ay0 + LABEL_HEIGHT + PADDING - THINK_PAD
    tx0 = ax0 + PADDING - THINK_PAD
    tx1 = ax1 - PADDING + THINK_PAD
    ty1 = ty0 + think_block_h
    draw.rounded_rectangle([tx0, ty0, tx1, ty1], radius=THINK_RADIUS, fill=THINK_BG, outline=THINK_BORDER, width=1)

    y = ty0 + THINK_PAD
    for line in think_lines:
        draw.text((ax0 + PADDING, y), line, fill='#222222', font=font)
        y += LINE_HEIGHT

    # Response text
    y = ty1 + LINE_HEIGHT
    for line in response_lines:
        draw.text((ax0 + PADDING, y), line, fill='#222222', font=font)
        y += LINE_HEIGHT

    img.save(outpath)
    print(f'Saved: {outpath} ({WIDTH}x{total_h})')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Render a single transcript as PNG')
    parser.add_argument('--user', required=True, help='User prompt text')
    parser.add_argument('--reasoning', required=True, help='Model reasoning/thinking text')
    parser.add_argument('--response', required=True, help='Model response text')
    parser.add_argument('--outpath', required=True, help='Output PNG path')
    args = parser.parse_args()
    render(args.user, args.reasoning, args.response, args.outpath)
