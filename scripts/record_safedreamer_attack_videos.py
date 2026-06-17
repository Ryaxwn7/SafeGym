import argparse
import datetime as dt
import json
import subprocess
import sys
import time
from pathlib import Path

from run_safedreamer_attack_suite import CONDITIONS, condition_names


def now():
    return dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')


def timestamp():
    return dt.datetime.now().strftime('%Y%m%d-%H%M%S')


def log(message, condition='videos'):
    print(f'[{now()}][{condition}] {message}', flush=True)


def build_eval_cmd(args, spec):
    cmd = [
        'python',
        'scripts/run_safedreamer_eval.py',
        '--steps',
        str(args.steps),
        '--logdir',
        str(args.outdir),
        '--condition',
        spec['condition'],
        '--attack',
        spec['attack'],
        '--attack-seed',
        str(args.seed),
        '--fast-eval',
        '--save-video',
    ]
    if args.checkpoint:
        cmd += ['--checkpoint', args.checkpoint]
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


def find_condition_videos(outdir, condition):
    condition_dir = Path(outdir) / condition
    if not condition_dir.exists():
        return []
    return sorted(condition_dir.rglob('*.mp4'), key=lambda path: path.stat().st_mtime)


def write_metadata(args, selected):
    metadata = {
        'created_at': now(),
        'outdir': str(args.outdir),
        'steps': args.steps,
        'seed': args.seed,
        'conditions': selected,
        'checkpoint': args.checkpoint,
    }
    path = Path(args.outdir) / 'video_run_metadata.json'
    path.write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    log(f'metadata: {path}')


def write_manifest(rows, outdir):
    path = Path(outdir) / 'video_manifest.csv'
    lines = ['condition,video_path']
    for row in rows:
        lines.append(f'{row["condition"]},{row["video_path"]}')
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    log(f'manifest: {path}')


def resolve_outdir(args):
    if args.outdir:
        return Path(args.outdir)
    run_id = args.run_id or timestamp()
    return Path(args.base_outdir) / run_id


def main():
    parser = argparse.ArgumentParser(
        description='Record SafeDreamer third-person videos for all attack conditions.'
    )
    parser.add_argument(
        '--base-outdir',
        default='results/safedreamer-video-runs',
        help='Base directory for timestamped video runs when --outdir is omitted.',
    )
    parser.add_argument('--run-id', default=None, help='Run folder name under --base-outdir.')
    parser.add_argument('--outdir', default=None, help='Explicit output directory.')
    parser.add_argument(
        '--steps',
        type=int,
        default=None,
        help='Raw SafeDreamer eval steps per condition. If omitted, computed from --episodes.',
    )
    parser.add_argument(
        '--episodes',
        type=int,
        default=1,
        help='Target complete episodes to record per condition. Default records one episode.',
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
        help='Extra eval steps to ensure the final episode video is emitted.',
    )
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--checkpoint', default=None, help='Optional checkpoint override.')
    parser.add_argument('--continue-on-error', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--list', action='store_true', help='print condition names and exit')
    parser.add_argument(
        '--only',
        nargs='*',
        default=None,
        help='optional subset of condition names, for example --only safedreamer_clean',
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

    args.outdir = resolve_outdir(args)
    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    log(f'outdir={args.outdir}')
    log(f'steps={args.steps} seed={args.seed} conditions={len(selected)}')
    if not args.dry_run:
        write_metadata(args, selected)

    rows = []
    failures = []
    for index, spec in enumerate(selected, start=1):
        condition = spec['condition']
        log(f'[{index}/{len(selected)}] start video recording attack={spec["attack"]}', condition)
        cmd = build_eval_cmd(args, spec)
        log_path = Path(args.outdir) / condition / 'video_run.log'
        if args.dry_run:
            log('dry-run: ' + ' '.join(cmd), condition)
            continue
        try:
            run_command(cmd, condition, log_path)
            videos = find_condition_videos(args.outdir, condition)
            if not videos:
                raise RuntimeError(f'{condition}: no mp4 video produced')
            for video in videos:
                rows.append({'condition': condition, 'video_path': str(video)})
            log(f'videos={len(videos)} latest={videos[-1]}', condition)
            write_manifest(rows, args.outdir)
        except Exception as exc:
            failures.append({'condition': condition, 'error': str(exc)})
            failure_path = Path(args.outdir) / 'video_failures.json'
            failure_path.write_text(json.dumps(failures, indent=2), encoding='utf-8')
            log(f'failed: {exc}', condition)
            if not args.continue_on_error:
                raise
            log('continuing after failure', condition)

    if not args.dry_run:
        write_manifest(rows, args.outdir)
    log('done')


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        log('interrupted by user')
        sys.exit(130)
