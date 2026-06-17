import argparse
from pathlib import Path

import pandas as pd


METRICS = ['avg_return', 'avg_true_cost', 'violation_rate']


def format_float(value):
    if pd.isna(value):
        return ''
    return f'{float(value):.4f}'


def build_comparison(old_path, new_path):
    old = pd.read_csv(old_path)
    new = pd.read_csv(new_path)
    old = old[['condition'] + [col for col in METRICS if col in old.columns]]
    new = new[['condition'] + [col for col in METRICS if col in new.columns]]
    merged = old.merge(new, on='condition', how='outer', suffixes=('_old', '_new'))

    for metric in METRICS:
        old_col = f'{metric}_old'
        new_col = f'{metric}_new'
        if old_col in merged.columns and new_col in merged.columns:
            merged[f'delta_{metric}'] = merged[new_col] - merged[old_col]

    ordered = ['condition']
    for metric in METRICS:
        ordered.extend([f'{metric}_old', f'{metric}_new', f'delta_{metric}'])
    ordered = [col for col in ordered if col in merged.columns]
    return merged[ordered].sort_values('condition')


def write_markdown(df, path, old_path, new_path):
    lines = [
        '# SafeDreamer Checkpoint Comparison',
        '',
        f'- Old summary: `{old_path}`',
        f'- New summary: `{new_path}`',
        '',
        '| condition | old return | new return | delta return | old cost | new cost | delta cost | old viol. | new viol. | delta viol. |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |',
    ]
    for _, row in df.iterrows():
        lines.append(
            '| '
            + ' | '.join(
                [
                    str(row['condition']),
                    format_float(row.get('avg_return_old')),
                    format_float(row.get('avg_return_new')),
                    format_float(row.get('delta_avg_return')),
                    format_float(row.get('avg_true_cost_old')),
                    format_float(row.get('avg_true_cost_new')),
                    format_float(row.get('delta_avg_true_cost')),
                    format_float(row.get('violation_rate_old')),
                    format_float(row.get('violation_rate_new')),
                    format_float(row.get('delta_violation_rate')),
                ]
            )
            + ' |'
        )
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--old', required=True, help='Previous checkpoint summary.csv.')
    parser.add_argument('--new', required=True, help='New checkpoint summary.csv.')
    parser.add_argument('--out-csv', required=True)
    parser.add_argument('--out-md', default=None)
    args = parser.parse_args()

    old_path = Path(args.old)
    new_path = Path(args.new)
    if not old_path.exists():
        raise SystemExit(f'Missing old summary: {old_path}')
    if not new_path.exists():
        raise SystemExit(f'Missing new summary: {new_path}')

    df = build_comparison(old_path, new_path)
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    if args.out_md:
        write_markdown(df, Path(args.out_md), old_path, new_path)

    print(df.to_string(index=False), flush=True)
    print(f'comparison_csv: {out_csv}', flush=True)
    if args.out_md:
        print(f'comparison_md: {args.out_md}', flush=True)


if __name__ == '__main__':
    main()
