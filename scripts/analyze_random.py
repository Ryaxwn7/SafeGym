import argparse
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', str(Path('.matplotlib-cache').resolve()))

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', default='results/random_clean.csv')
    parser.add_argument('--out', default='figures/random_clean_summary.png')
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    avg_return = float(df['return'].mean())
    avg_cost = float(df['cost'].mean())
    violation_rate = float(df['violated'].mean())

    print('avg_return:', avg_return)
    print('avg_cost:', avg_cost)
    print('violation_rate:', violation_rate)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(['avg_return', 'avg_cost', 'violation_rate'], [avg_return, avg_cost, violation_rate])
    ax.set_title('Random Baseline Summary')
    ax.grid(axis='y', alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)

    print('figure:', out_path)


if __name__ == '__main__':
    main()
