"""Render SFT transcript .txt files as PNGs using render_transcript_single.

Parses the structured .txt format (header / USER PROMPT / REASONING / RESPONSE)
and delegates to render_transcript_single.render().

For transcripts containing CJK text, use --cjk-font to use a CJK-capable
system font (tries Hiragino Sans GB, STHeiti Medium on macOS).

Usage:
    python scripts/analysis/render_transcript_sft.py \
        results/summaries/transcripts/2026_03_28/sft_gpt-oss-120b_english_capital.txt

    python scripts/analysis/render_transcript_sft.py --cjk-font \
        results/summaries/transcripts/2026_03_28/sft_qwen3-32b_reasoning_language_zh.txt
"""
import argparse
import re
import sys
from pathlib import Path

# Add scripts/analysis to path so we can import the renderer
sys.path.insert(0, str(Path(__file__).resolve().parent))
import render_transcript_single as renderer
from PIL import ImageFont


def _is_list_item(line: str) -> bool:
    """Check if a line starts a list item (- bullet or 1. numbered)."""
    s = line.lstrip()
    return s.startswith('- ') or bool(re.match(r'\d+[\.\)]\s', s))


def unwrap(text: str) -> str:
    """Rejoin hard-wrapped lines into paragraphs.

    Consecutive non-blank lines are joined (no space for CJK, space for Latin).
    Blank lines and list items are preserved as paragraph separators.
    Lines starting with - or N. are kept as separate paragraphs.
    Indented continuation lines (e.g. "   and the Venus de Milo.") are joined
    to the previous line.
    """
    paragraphs = []
    current = []

    def flush():
        if current:
            paragraphs.append(''.join(current))
            current.clear()

    for line in text.split('\n'):
        if line.strip() == '':
            flush()
            paragraphs.append('')
        elif _is_list_item(line):
            # Start a new paragraph for each list item
            flush()
            current.append(line)
        elif current and line.startswith('   '):
            # Indented continuation of a list item — join with space
            current.append(' ' + line.strip())
        else:
            if current:
                prev = current[-1]
                if (prev and ord(prev[-1]) > 0x2E80) or (line and ord(line[0]) > 0x2E80):
                    current.append(line)
                else:
                    current.append(' ' + line)
            else:
                current.append(line)
    flush()
    return '\n'.join(paragraphs)


def parse_transcript_txt(path: str) -> dict:
    """Parse a structured SFT transcript .txt file into sections."""
    text = Path(path).read_text()

    # Split on the ===... section headers
    sections = re.split(r'={50,}\n(.+?)\n={50,}\n', text)
    # sections[0] = header, then alternating (section_name, section_body)

    result = {}
    for i in range(1, len(sections), 2):
        name = sections[i].strip()
        body = sections[i + 1].strip() if i + 1 < len(sections) else ''
        result[name] = body

    return result


def main():
    parser = argparse.ArgumentParser(description='Render SFT transcript .txt as PNG')
    parser.add_argument('txt_files', nargs='+', help='Transcript .txt file(s)')
    parser.add_argument('--cjk-font', action='store_true',
                        help='Use PingFang HK for CJK text support')
    parser.add_argument('--outdir', default=None,
                        help='Output directory (default: same as input)')
    args = parser.parse_args()

    if args.cjk_font:
        CJK_CANDIDATES = [
            '/System/Library/Fonts/Hiragino Sans GB.ttc',
            '/System/Library/Fonts/STHeiti Medium.ttc',
            '/System/Library/Fonts/PingFang.ttc',
            '/Library/Fonts/Microsoft/SimHei.ttf',
        ]
        loaded = False
        for fpath in CJK_CANDIDATES:
            try:
                cjk = ImageFont.truetype(fpath, renderer.FONT_SIZE)
                renderer.font = cjk
                renderer.WIDTH = 1000
                loaded = True
                break
            except Exception:
                continue
        if not loaded:
            print('Warning: no CJK font found, Chinese text may not render')

        # Width-aware wrap: CJK chars count as 2 columns, Latin as 1
        import unicodedata

        def _display_width(ch):
            ea = unicodedata.east_asian_width(ch)
            return 2 if ea in ('W', 'F') else 1

        def _cjk_aware_wrap(text, width=80):
            lines = []
            for para in text.split('\n'):
                if para.strip() == '':
                    lines.append('')
                    continue
                # Manual wrap respecting display widths
                line = ''
                col = 0
                for ch in para:
                    w = _display_width(ch)
                    if col + w > width:
                        lines.append(line)
                        line = ch
                        col = w
                    else:
                        line += ch
                        col += w
                if line:
                    lines.append(line)
            return lines

        renderer.wrap_text = _cjk_aware_wrap

    for txt_path in args.txt_files:
        sections = parse_transcript_txt(txt_path)

        user_text = unwrap(sections.get('USER PROMPT', ''))
        # The reasoning section name varies (e.g. "REASONING (STRIPPED — ...)")
        reasoning = ''
        for key in sections:
            if key.startswith('REASONING'):
                reasoning = unwrap(sections[key])
                break
        response = unwrap(sections.get('RESPONSE', ''))

        outdir = args.outdir or str(Path(txt_path).parent)
        outpath = str(Path(outdir) / (Path(txt_path).stem + '.png'))

        renderer.render(user_text, reasoning, response, outpath)


if __name__ == '__main__':
    main()
