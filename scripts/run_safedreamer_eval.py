import argparse
import os
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--logdir', default='results/2026-06-13-safedreamer-eval')
    parser.add_argument(
        '--checkpoint',
        default=(
            'external/checkpoints/safedreamer_osrp_vector/safedreamer_osrp_vector/'
            '20240307-010600_osrp_vector_safetygymcoor_SafetyPointGoal1-v0_0.ckpt'
        ),
    )
    args = parser.parse_args()

    Path(args.logdir).mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.setdefault('MUJOCO_GL', 'egl')

    cmd = [
        'conda',
        'run',
        '-n',
        'safedreamer-py38',
        'python',
        'external/SafeDreamer/SafeDreamer/train.py',
        '--configs',
        'osrp_vector',
        '--run.script',
        'eval_only',
        '--run.from_checkpoint',
        args.checkpoint,
        '--task',
        'safetygymcoor_SafetyPointGoal1-v0',
        '--jax.platform',
        'cpu',
        '--run.steps',
        str(args.steps),
        '--logdir',
        args.logdir + '/',
    ]
    print(' '.join(cmd))
    subprocess.run(cmd, check=True, env=env)


if __name__ == '__main__':
    main()
