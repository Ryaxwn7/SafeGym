import argparse
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', str(Path('.matplotlib-cache').resolve()))

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import pandas as pd


def condition_name(path):
    name = Path(path).stem
    prefix = 'random_'
    if name.startswith(prefix):
        return name[len(prefix) :]
    return name


def summarize(path):
    df = pd.read_csv(path)
    true_cost = df['true_cost'] if 'true_cost' in df.columns else df['cost']
    exposed_cost = df['exposed_cost'] if 'exposed_cost' in df.columns else df['cost']
    return {
        'condition': condition_name(path),
        'file': Path(path).name,
        'episodes': len(df),
        'avg_return': float(df['return'].mean()),
        'avg_cost': float(df['cost'].mean()),
        'avg_true_cost': float(true_cost.mean()),
        'avg_exposed_cost': float(exposed_cost.mean()),
        'violation_rate': float(df['violated'].mean()),
        'avg_steps': float(df['steps'].mean()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputs', nargs='+', required=True)
    parser.add_argument('--outdir', default='results')
    parser.add_argument('--figdir', default='figures')
    args = parser.parse_args()

    outdir = Path(args.outdir)
    figdir = Path(args.figdir)
    outdir.mkdir(parents=True, exist_ok=True)
    figdir.mkdir(parents=True, exist_ok=True)

    summary = pd.DataFrame([summarize(path) for path in args.inputs])
    summary_path = outdir / 'summary.csv'
    summary.to_csv(summary_path, index=False)
    print(summary.to_string(index=False))
    print('summary:', summary_path)

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8))
    metrics = [
        ('avg_return', 'Average return'),
        ('avg_true_cost', 'Average true cost'),
        ('violation_rate', 'Violation rate'),
    ]
    colors = ['#4c78a8', '#f58518', '#54a24b']

    for ax, (metric, title) in zip(axes, metrics):
        ax.bar(summary['condition'], summary[metric], color=colors)
        ax.set_title(title)
        ax.tick_params(axis='x', rotation=20)
        ax.grid(axis='y', alpha=0.25)

    fig.suptitle('Safety-Gymnasium random policy under perturbations')
    fig.tight_layout()
    fig_path = figdir / 'summary_metrics.png'
    fig.savefig(fig_path, dpi=180)
    plt.close(fig)
    print('figure:', fig_path)


if __name__ == '__main__':
    main()
