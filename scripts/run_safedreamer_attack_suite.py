import argparse
import datetime as dt
import json
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd


CONDITIONS = [
    {'condition': 'safedreamer_clean', 'attack': 'none'},
    {'condition': 'safedreamer_hazard_blind', 'attack': 'hazard_blind'},
    {'condition': 'safedreamer_lidar_blind', 'attack': 'lidar_blind', 'attack_strength': 8},
    {'condition': 'safedreamer_obs_noise_0.1', 'attack': 'obs_noise', 'obs_noise_std': 0.1},
    {'condition': 'safedreamer_obs_delay_2', 'attack': 'obs_delay', 'obs_delay': 2},
    {'condition': 'safedreamer_cost_under', 'attack': 'cost_under', 'cost_scale': 0.25},
    {'condition': 'safedreamer_action_noise_0.1', 'attack': 'action_noise', 'action_noise_std': 0.1},
    {'condition': 'safedreamer_action_delay_2', 'attack': 'action_delay', 'action_delay': 2},
]


def now():
    return dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')


def timestamp():
    return dt.datetime.now().strftime('%Y%m%d-%H%M%S')


def log(message, condition='suite'):
    print(f'[{now()}][{condition}] {message}', flush=True)


def condition_names():
    return [item['condition'] for item in CONDITIONS]


def run_command(cmd, condition, log_path):
    started = time.time()
    command_text = ' '.join(str(part) for part in cmd)
    log(f'command: {command_text}', condition)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open('a', encoding='utf-8') as file:
        file.write(f'\n\n[{now()}] COMMAND {command_text}\n')
        file.flush()
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            clean = line.rstrip()
            if clean:
                print(f'[{condition}] {clean}', flush=True)
            file.write(line)
            file.flush()
        returncode = proc.wait()
        elapsed = time.time() - started
        file.write(f'[{now()}] EXIT {returncode} elapsed_sec={elapsed:.1f}\n')
    if returncode:
        raise subprocess.CalledProcessError(returncode, cmd)
    log(f'completed in {elapsed:.1f}s', condition)


def build_eval_cmd(args, spec):
    cmd = [
        'python',
        'scripts/run_safedreamer_eval.py',
        '--steps',
        str(args.steps),
        '--logdir',
        args.outdir,
        '--condition',
        spec['condition'],
        '--attack',
        spec['attack'],
        '--attack-seed',
        str(args.seed),
        '--fast-eval',
    ]
    for key in [
        'attack_strength',
        'obs_noise_std',
        'obs_delay',
        'action_noise_std',
        'action_delay',
        'cost_scale',
    ]:
        if key in spec:
            cmd += ['--' + key.replace('_', '-'), str(spec[key])]
    return cmd


def collect_cmd(args, condition, out_csv):
    return [
        'conda',
        'run',
        '-n',
        'safegym',
        'python',
        'scripts/collect_safedreamer_results.py',
        '--root',
        str(Path(args.outdir) / condition),
        '--condition',
        condition,
        '--out',
        str(out_csv),
    ]


def validate_csv(path, condition, expected_episode_length=None, allow_short_episodes=False):
    if not path.exists() or path.stat().st_size == 0:
        raise RuntimeError(f'{condition}: missing CSV {path}')
    df = pd.read_csv(path)
    if df.empty:
        raise RuntimeError(f'{condition}: empty CSV {path}')
    required = {'condition', 'return', 'true_cost', 'exposed_cost', 'violated'}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f'{condition}: CSV missing columns {sorted(missing)}')
    if expected_episode_length is not None and not allow_short_episodes:
        bad = df.loc[df['steps'] != expected_episode_length]
        if not bad.empty:
            observed = sorted(set(int(value) for value in bad['steps'].tolist()))
            raise RuntimeError(
                f'{condition}: expected episode length {expected_episode_length}, '
                f'observed {observed}')
    log(
        'csv ok: '
        f'episodes={len(df)} '
        f'avg_return={df["return"].mean():.4f} '
        f'avg_true_cost={df["true_cost"].mean():.4f} '
        f'violation_rate={df["violated"].mean():.4f}',
        condition,
    )
    return df


def run_condition(args, spec, index, total):
    condition = spec['condition']
    outdir = Path(args.outdir)
    out_csv = outdir / f'{condition}.csv'
    log_path = outdir / condition / 'run.log'
    log(f'[{index}/{total}] start attack={spec["attack"]}', condition)

    if out_csv.exists() and out_csv.stat().st_size > 0 and args.resume and not args.force:
        log(f'skip existing CSV: {out_csv}', condition)
        validate_csv(out_csv, condition, args.episode_length, args.allow_short_episodes)
        return out_csv
    if out_csv.exists() and out_csv.stat().st_size > 0 and not args.force:
        raise RuntimeError(
            f'{condition}: CSV already exists at {out_csv}. '
            'Use --resume to skip completed conditions, --force to overwrite, '
            'or omit --outdir to create a new timestamped run directory.')

    eval_cmd = build_eval_cmd(args, spec)
    if args.dry_run:
        log('dry-run eval: ' + ' '.join(eval_cmd), condition)
        log('dry-run collect: ' + ' '.join(str(x) for x in collect_cmd(args, condition, out_csv)), condition)
        return out_csv

    for attempt in range(1, args.retries + 2):
        try:
            log(f'eval attempt {attempt}/{args.retries + 1}', condition)
            run_command(eval_cmd, condition, log_path)
            log('collecting scores.jsonl into CSV', condition)
            run_command(collect_cmd(args, condition, out_csv), condition, log_path)
            validate_csv(out_csv, condition, args.episode_length, args.allow_short_episodes)
            return out_csv
        except Exception as exc:
            log(f'failed attempt {attempt}: {exc}', condition)
            if attempt > args.retries:
                raise
            time.sleep(args.retry_sleep)
    return out_csv


def summarize(csv_paths, outdir):
    frames = [pd.read_csv(path) for path in csv_paths if path.exists() and path.stat().st_size > 0]
    if not frames:
        raise RuntimeError('No SafeDreamer attack CSV files were produced.')
    all_df = pd.concat(frames, ignore_index=True)
    all_path = Path(outdir) / 'all_episodes.csv'
    all_df.to_csv(all_path, index=False)

    summary = (
        all_df.groupby('condition', as_index=False)
        .agg(
            episodes=('episode', 'count'),
            avg_return=('return', 'mean'),
            avg_cost=('cost', 'mean'),
            avg_true_cost=('true_cost', 'mean'),
            avg_exposed_cost=('exposed_cost', 'mean'),
            violation_rate=('violated', 'mean'),
        )
        .sort_values('condition')
    )
    clean = summary.loc[summary['condition'] == 'safedreamer_clean']
    clean_cost = float(clean['avg_true_cost'].iloc[0]) if not clean.empty else 0.0
    clean_violation = float(clean['violation_rate'].iloc[0]) if not clean.empty else 0.0
    summary['delta_true_cost_vs_clean'] = summary['avg_true_cost'] - clean_cost
    summary['delta_violation_vs_clean'] = summary['violation_rate'] - clean_violation

    summary_path = Path(outdir) / 'summary.csv'
    summary.to_csv(summary_path, index=False)
    log(f'all episodes: {all_path}')
    log(f'summary: {summary_path}')
    print(summary.to_string(index=False), flush=True)


def write_failures(failures, outdir):
    if not failures:
        return
    path = Path(outdir) / 'failures.csv'
    pd.DataFrame(failures).to_csv(path, index=False)
    log(f'failures: {path}')


def resolve_outdir(args):
    if args.outdir:
        return Path(args.outdir)
    run_id = args.run_id or timestamp()
    return Path(args.base_outdir) / run_id


def write_run_metadata(args, selected, outdir):
    metadata = {
        'created_at': now(),
        'outdir': str(outdir),
        'steps': args.steps,
        'episodes': args.episodes,
        'episode_length': args.episode_length,
        'step_margin': args.step_margin,
        'seed': args.seed,
        'resume': args.resume,
        'force': args.force,
        'conditions': selected,
    }
    path = Path(outdir) / 'run_metadata.json'
    path.write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    log(f'metadata: {path}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--base-outdir',
        default='results/safedreamer-attack-runs',
        help='Base directory for timestamped runs when --outdir is not provided.',
    )
    parser.add_argument(
        '--run-id',
        default=None,
        help='Run folder name under --base-outdir. Defaults to current timestamp.',
    )
    parser.add_argument(
        '--outdir',
        default=None,
        help='Explicit output directory. If omitted, a timestamped directory is created.',
    )
    parser.add_argument(
        '--steps',
        type=int,
        default=None,
        help='Raw SafeDreamer eval steps. If omitted, computed from --episodes.',
    )
    parser.add_argument(
        '--episodes',
        type=int,
        default=1,
        help='Target complete episodes per condition. Default 1 for a full smoke suite.',
    )
    parser.add_argument(
        '--episode-length',
        type=int,
        default=200,
        help='Expected SafeDreamer policy steps per episode. SafetyPointGoal1-v0 is 200 with repeat=5.',
    )
    parser.add_argument(
        '--step-margin',
        type=int,
        default=50,
        help='Extra eval steps to ensure the final episode is emitted.',
    )
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--retries', type=int, default=1)
    parser.add_argument('--retry-sleep', type=float, default=5.0)
    parser.add_argument('--resume', action='store_true', help='skip existing per-condition CSV files in --outdir')
    parser.add_argument('--force', action='store_true', help='rerun conditions even if CSV exists')
    parser.add_argument('--continue-on-error', action='store_true')
    parser.add_argument(
        '--allow-short-episodes',
        action='store_true',
        help='Do not fail if an environment terminates before --episode-length.',
    )
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--list', action='store_true', help='print condition names and exit')
    parser.add_argument(
        '--only',
        nargs='*',
        default=None,
        help='optional subset of condition names, for example --only safedreamer_clean safedreamer_cost_under',
    )
    args = parser.parse_args()

    if args.list:
        print('\n'.join(condition_names()))
        return

    if args.steps is None:
        args.steps = args.episodes * args.episode_length + args.step_margin

    selected = CONDITIONS
    if args.only:
        wanted = set(args.only)
        selected = [item for item in CONDITIONS if item['condition'] in wanted]
        missing = wanted - {item['condition'] for item in selected}
        if missing:
            raise SystemExit(f'Unknown conditions: {sorted(missing)}')

    args.outdir = str(resolve_outdir(args))
    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    log(f'outdir={args.outdir}')
    log(f'steps={args.steps} seed={args.seed} conditions={len(selected)}')
    if not args.dry_run:
        write_run_metadata(args, selected, args.outdir)

    csv_paths = []
    failures = []
    for index, spec in enumerate(selected, start=1):
        condition = spec['condition']
        try:
            csv_paths.append(run_condition(args, spec, index, len(selected)))
        except Exception as exc:
            failures.append({'condition': condition, 'error': str(exc)})
            write_failures(failures, args.outdir)
            if not args.continue_on_error:
                raise
            log(f'continuing after failure: {condition}', condition)

    if not args.dry_run:
        summarize(csv_paths, args.outdir)
        write_failures(failures, args.outdir)
    log('done')


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        log('interrupted by user')
        sys.exit(130)
