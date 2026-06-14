import argparse
import os
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--logdir', default='results/2026-06-13-safedreamer-eval')
    parser.add_argument('--condition', default='safedreamer_clean')
    parser.add_argument(
        '--attack',
        default='none',
        choices=[
            'none',
            'hazard_blind',
            'lidar_blind',
            'obs_noise',
            'obs_delay',
            'cost_under',
            'action_noise',
            'action_delay',
        ],
    )
    parser.add_argument('--attack-strength', type=float, default=0.0)
    parser.add_argument('--obs-noise-std', type=float, default=0.0)
    parser.add_argument('--obs-delay', type=int, default=0)
    parser.add_argument('--action-noise-std', type=float, default=0.0)
    parser.add_argument('--action-delay', type=int, default=0)
    parser.add_argument('--cost-scale', type=float, default=1.0)
    parser.add_argument('--attack-seed', type=int, default=0)
    parser.add_argument('--fast-eval', action='store_true')
    parser.add_argument(
        '--checkpoint',
        default=(
            'external/checkpoints/safedreamer_osrp_vector/safedreamer_osrp_vector/'
            '20240307-010600_osrp_vector_safetygymcoor_SafetyPointGoal1-v0_0.ckpt'
        ),
    )
    args = parser.parse_args()

    logdir = Path(args.logdir) / args.condition
    logdir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.setdefault('MUJOCO_GL', 'egl')
    env.setdefault('MPLCONFIGDIR', '/tmp/matplotlib-safedreamer')
    if args.fast_eval:
        env['SAFEDREAMER_FAST_EVAL'] = '1'

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
        str(logdir) + '/',
        '--env.safetygymcoor.attack',
        args.attack,
        '--env.safetygymcoor.attack_strength',
        str(args.attack_strength),
        '--env.safetygymcoor.obs_noise_std',
        str(args.obs_noise_std),
        '--env.safetygymcoor.obs_delay',
        str(args.obs_delay),
        '--env.safetygymcoor.action_noise_std',
        str(args.action_noise_std),
        '--env.safetygymcoor.action_delay',
        str(args.action_delay),
        '--env.safetygymcoor.cost_scale',
        str(args.cost_scale),
        '--env.safetygymcoor.attack_seed',
        str(args.attack_seed),
    ]
    print(' '.join(cmd))
    subprocess.run(cmd, check=True, env=env)


if __name__ == '__main__':
    main()
