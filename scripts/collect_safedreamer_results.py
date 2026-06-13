import argparse
import json
from pathlib import Path

import pandas as pd


def latest_logdir(root):
    candidates = [
        path
        for path in Path(root).iterdir()
        if path.is_dir() and (path / 'scores.jsonl').exists() and (path / 'scores.jsonl').stat().st_size > 0
    ]
    if not candidates:
        raise FileNotFoundError(f'No SafeDreamer logdirs found under {root}')
    return max(candidates, key=lambda path: path.stat().st_mtime)


def read_scores(path):
    rows = []
    episode = 0
    pending = {}
    with Path(path).open('r', encoding='utf-8') as file:
        for line in file:
            if not line.strip():
                continue
            item = json.loads(line)
            if 'episode/score' in item:
                pending['return'] = float(item['episode/score'])
            if 'episode/cost' in item:
                pending['cost'] = float(item['episode/cost'])
            if 'episode/score' in item or 'episode/cost' in item:
                pending['step'] = int(item.get('step', 0))
            if 'return' in pending and 'cost' in pending:
                rows.append(
                    {
                        'condition': 'safedreamer_clean',
                        'episode': episode,
                        'step': pending.get('step', 0),
                        'return': pending['return'],
                        'cost': pending['cost'],
                        'true_cost': pending['cost'],
                        'exposed_cost': pending['cost'],
                        'steps': 200,
                        'violated': int(pending['cost'] > 0.0),
                    }
                )
                episode += 1
                pending = {}
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='results/2026-06-13-safedreamer-eval')
    parser.add_argument('--out', default='results/2026-06-13-safedreamer-eval/safedreamer_clean.csv')
    args = parser.parse_args()

    logdir = latest_logdir(args.root)
    scores_path = logdir / 'scores.jsonl'
    if not scores_path.exists():
        raise FileNotFoundError(scores_path)

    rows = read_scores(scores_path)
    if not rows:
        raise RuntimeError(f'No episode score/cost rows found in {scores_path}')

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)

    print('logdir:', logdir)
    print('csv:', out_path)
    print('episodes:', len(df))
    print('avg_return:', float(df['return'].mean()))
    print('avg_true_cost:', float(df['true_cost'].mean()))
    print('violation_rate:', float(df['violated'].mean()))


if __name__ == '__main__':
    main()
