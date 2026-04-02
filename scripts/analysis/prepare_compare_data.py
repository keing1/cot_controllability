"""Extract base-vs-FT transcript pairs from rollout JSONL files into a JSON
file suitable for render_transcript_compare.py.

Usage:
    python scripts/analysis/prepare_compare_data.py \
        --output /tmp/compare_data.json

    # Then render:
    python scripts/analysis/render_transcript_compare.py \
        --data /tmp/compare_data.json

Edit the `PAIRS` list below to select which samples to compare.
"""
import argparse
import json


# ── Configure which samples to compare ──
# Each entry specifies a base/FT rollout file pair and which (sample_id, control_mode) to extract.
PAIRS = [
    {
        "label": "GPT-OSS-120B",
        "sid": "cotcontrol/gpqa_119", "cm": "word_suppression",
        "base_file": "results/rollouts/openai_gpt-oss-120b_cotcontrol_all_913928bcdb93.jsonl",
        "ft_file": "results/rollouts/openai_gpt-oss-120b-rif-lr1e-4-000060_cotcontrol_all_20260319_093421_0bb4a3a28459.jsonl",
        "out": "results/summaries/transcripts/compare_gpt-oss-120b_cotcontrol_gpqa_119_word_suppression.png",
    },
    {
        "label": "Qwen3-8B",
        "sid": "reasonif_262", "cm": "end_checker",
        "base_file": "results/rollouts/qwen_qwen3-8b_reasonif_all_bdb6aebc71c4.jsonl",
        "ft_file": "results/rollouts/qwen_qwen3-8b-rif-lr1e-4-000060_reasonif_all_20260319_093421_e94ef7e84924.jsonl",
        "out": "results/summaries/transcripts/compare_qwen3-8b_reasonif_262_end_checker.png",
    },
    {
        "label": "GPT-OSS-120B",
        "sid": "reasonif_85", "cm": "english_capital",
        "base_file": "results/rollouts/openai_gpt-oss-120b_reasonif_all_462e3cde6aec.jsonl",
        "ft_file": "results/rollouts/openai_gpt-oss-120b-rif-lr1e-4-000060_reasonif_all_20260319_093421_a5bf3180e74e.jsonl",
        "out": "results/summaries/transcripts/compare_gpt-oss-120b_reasonif_85_english_capital.png",
    },
    {
        "label": "Qwen3-32B",
        "sid": "cotcontrol/gpqa_149", "cm": "ignore_question",
        "base_file": "results/rollouts/qwen_qwen3-32b_cotcontrol_all_135ddb8ed65f.jsonl",
        "ft_file": "results/rollouts/qwen_qwen3-32b-rif-lr1e-4-000060_cotcontrol_all_20260319_093421_e4737ce08cd1.jsonl",
        "out": "results/summaries/transcripts/compare_qwen3-32b_cotcontrol_gpqa_149_ignore_question.png",
    },
]

# ── repeat_sentences pairs (GPT-OSS-20B) ──
REPEAT_PAIRS = {
    "base_file": "results/rollouts/openai_gpt-oss-20b_cotcontrol_all_c9e55b1ebb5f.jsonl",
    "ft_file": "results/rollouts/openai_gpt-oss-20b-rif-lr1e-4-000060_cotcontrol_all_20260319_093421_e8e14135d65a.jsonl",
    "sids": [
        "cotcontrol/gpqa_48", "cotcontrol/gpqa_376", "cotcontrol/gpqa_119",
        "cotcontrol/gpqa_45", "cotcontrol/gpqa_363", "cotcontrol/gpqa_323",
    ],
}


def find_record(fpath, sid, cm):
    with open(fpath) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            r = json.loads(line)
            if r.get('sample', {}).get('id') == sid and r.get('control_mode') == cm:
                return r
    return None


def load_rollouts_by_mode(path, control_mode):
    rollouts = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            obj = json.loads(line)
            if obj.get("control_mode") == control_mode:
                sid = obj["sample"]["id"]
                rollouts[sid] = obj
    return rollouts


def build_main_pairs(output_path):
    items = []
    for p in PAIRS:
        base = find_record(p['base_file'], p['sid'], p['cm'])
        ft = find_record(p['ft_file'], p['sid'], p['cm'])
        print(f"{p['label']} | {p['sid']} | {p['cm']}")
        print(f"  base compliant={base.get('compliant') if base else 'NOT FOUND'}")
        print(f"  ft   compliant={ft.get('compliant') if ft else 'NOT FOUND'}")
        items.append({
            'label': p['label'],
            'sid': p['sid'],
            'cm': p['cm'],
            'outpath': p['out'],
            'base_user_prompt': base.get('user_prompt', '') if base else '',
            'base_reasoning': base.get('reasoning', '') if base else '',
            'base_response': base.get('response', '') if base else '',
            'base_compliant': base.get('compliant') if base else None,
            'ft_reasoning': ft.get('reasoning', '') if ft else '',
            'ft_response': ft.get('response', '') if ft else '',
            'ft_compliant': ft.get('compliant') if ft else None,
        })
    with open(output_path, 'w') as f:
        json.dump(items, f)
    print(f'\nSaved {len(items)} pairs to {output_path}')


def build_repeat_pairs(output_path):
    rp = REPEAT_PAIRS
    base = load_rollouts_by_mode(rp["base_file"], "repeat_sentences")
    ft = load_rollouts_by_mode(rp["ft_file"], "repeat_sentences")
    items = []
    for sid in rp["sids"]:
        b = base[sid]
        f_ = ft[sid]
        short_sid = sid.replace("cotcontrol/", "")
        items.append({
            "label": "GPT-OSS-20B",
            "sid": sid,
            "cm": "repeat_sentences",
            "outpath": f"results/summaries/transcripts/compare_gpt-oss-20b_{short_sid}_repeat_sentences.png",
            "base_user_prompt": b["user_prompt"],
            "base_reasoning": b.get("reasoning", "") or "",
            "base_response": b["response"],
            "base_compliant": b["compliant"],
            "ft_reasoning": f_.get("reasoning", "") or "",
            "ft_response": f_["response"],
            "ft_compliant": f_["compliant"],
        })
        print(f"{sid}: base_compliant={b['compliant']}, ft_compliant={f_['compliant']}")
    with open(output_path, 'w') as f_out:
        json.dump(items, f_out)
    print(f'\nSaved {len(items)} repeat_sentences pairs to {output_path}')


def main():
    parser = argparse.ArgumentParser(description='Extract base-vs-FT transcript pairs')
    parser.add_argument('--output', default='/tmp/compare_data.json',
                        help='Output JSON path for main comparisons')
    parser.add_argument('--repeat-output', default='/tmp/compare_data_repeat.json',
                        help='Output JSON path for repeat_sentences comparisons')
    parser.add_argument('--mode', choices=['main', 'repeat', 'both'], default='both',
                        help='Which pairs to extract')
    args = parser.parse_args()

    if args.mode in ('main', 'both'):
        build_main_pairs(args.output)
    if args.mode in ('repeat', 'both'):
        build_repeat_pairs(args.repeat_output)


if __name__ == '__main__':
    main()
