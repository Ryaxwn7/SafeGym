import argparse
import datetime as dt
import json
import subprocess
import sys
import time
from pathlib import Path


DEFAULT_CHECKPOINT = (
    'external/checkpoints/safedreamer_osrp_vector/safedreamer_osrp_vector/checkpoint.ckpt'
)
DEFAULT_BASELINE = 'results/safedreamer-attack-runs/20260616-112213/summary.csv'


def now():
    return dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')


def timestamp():
    return dt.datetime.now().strftime('%Y%m%d-%H%M%S')


def log(message):
    print(f'[{now()}][costlimit25] {message}', flush=True)


def run_live(cmd, dry_run=False):
    command_text = ' '.join(str(part) for part in cmd)
    log(f'command: {command_text}')
    if dry_run:
        return 0

    started = time.time()
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end='', flush=True)
    code = proc.wait()
    elapsed = time.time() - started
    log(f'exit={code} elapsed_sec={elapsed:.1f}')
    return code


def append_only(cmd, only):
    if only:
        cmd += ['--only'] + only
    return cmd


def main():
    parser = argparse.ArgumentParser(
        description='Run costlimit=25 SafeDreamer attack evaluation, comparison, and videos.'
    )
    parser.add_argument('--checkpoint', default=DEFAULT_CHECKPOINT)
    parser.add_argument('--baseline-summary', default=DEFAULT_BASELINE)
    parser.add_argument('--episodes', type=int, default=5)
    parser.add_argument('--episode-length', type=int, default=200)
    parser.add_argument('--step-margin', type=int, default=50)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--run-label', default='costlimit25')
    parser.add_argument('--skip-eval', action='store_true')
    parser.add_argument('--skip-videos', action='store_true')
    parser.add_argument('--continue-on-error', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument(
        '--only',
        nargs='*',
        default=None,
        help='Optional subset of conditions, for example --only safedreamer_clean.',
    )
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        raise SystemExit(f'Missing checkpoint: {checkpoint}')

    run_id = f'{args.run_label}-{timestamp()}'
    eval_dir = Path('results/safedreamer-attack-runs') / run_id
    video_dir = Path('results/safedreamer-video-runs') / run_id
    package_dir = Path('results/costlimit25-full-runs') / run_id

    manifest = {
        'created_at': now(),
        'run_id': run_id,
        'checkpoint': str(checkpoint),
        'baseline_summary': args.baseline_summary,
        'episodes': args.episodes,
        'episode_length': args.episode_length,
        'step_margin': args.step_margin,
        'seed': args.seed,
        'eval_dir': str(eval_dir),
        'video_dir': str(video_dir),
        'comparison_csv': str(package_dir / 'comparison_vs_previous.csv'),
        'comparison_md': str(package_dir / 'comparison_vs_previous.md'),
        'only': args.only,
    }
    if not args.dry_run:
        package_dir.mkdir(parents=True, exist_ok=True)
        (package_dir / 'run_manifest.json').write_text(
            json.dumps(manifest, indent=2),
            encoding='utf-8',
        )

    log(f'run_id={run_id}')
    log(f'checkpoint={checkpoint}')
    log(f'eval_dir={eval_dir}')
    log(f'video_dir={video_dir}')
    log(f'package_dir={package_dir}')

    common = [
        '--episodes',
        str(args.episodes),
        '--episode-length',
        str(args.episode_length),
        '--step-margin',
        str(args.step_margin),
        '--seed',
        str(args.seed),
        '--checkpoint',
        str(checkpoint),
        '--run-id',
        run_id,
    ]
    if args.continue_on_error:
        common.append('--continue-on-error')

    if not args.skip_eval:
        cmd = ['python', 'scripts/run_safedreamer_attack_suite.py'] + common
        append_only(cmd, args.only)
        code = run_live(cmd, args.dry_run)
        if code:
            raise SystemExit(code)

    summary = eval_dir / 'summary.csv'
    baseline = Path(args.baseline_summary)
    if not args.skip_eval and summary.exists() and baseline.exists():
        compare_cmd = [
            'python',
            'scripts/compare_safedreamer_summaries.py',
            '--old',
            str(baseline),
            '--new',
            str(summary),
            '--out-csv',
            str(package_dir / 'comparison_vs_previous.csv'),
            '--out-md',
            str(package_dir / 'comparison_vs_previous.md'),
        ]
        code = run_live(compare_cmd, args.dry_run)
        if code:
            raise SystemExit(code)
    elif not args.skip_eval:
        log(f'skip comparison; missing baseline or new summary: {baseline}, {summary}')

    if not args.skip_videos:
        cmd = ['python', 'scripts/record_safedreamer_attack_videos.py'] + common
        append_only(cmd, args.only)
        code = run_live(cmd, args.dry_run)
        if code:
            raise SystemExit(code)

    log('done')
    if args.dry_run:
        log('dry-run completed without writing result files')
    else:
        log(f'manifest: {package_dir / "run_manifest.json"}')


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        log('interrupted by user')
        sys.exit(130)
