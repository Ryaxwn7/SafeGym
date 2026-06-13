import argparse
import os
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import safety_gymnasium
from PIL import Image, ImageDraw
from stable_baselines3 import PPO


def annotate(frame, step, reward, total_reward, cost, total_cost):
    image = Image.fromarray(np.asarray(frame, dtype=np.uint8))
    draw = ImageDraw.Draw(image)
    lines = [
        f'PPO step={step}',
        f'reward={reward:.3f} total={total_reward:.3f}',
        f'cost={cost:.1f} total={total_cost:.1f}',
    ]
    x, y = 8, 8
    for line in lines:
        draw.text((x + 1, y + 1), line, fill=(0, 0, 0))
        draw.text((x, y), line, fill=(255, 255, 255))
        y += 14
    return np.asarray(image)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='models/ppo_safety_point_goal_reward_only_100k.zip')
    parser.add_argument('--env', default='SafetyPointGoal1-v0')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--frame-skip', type=int, default=5)
    parser.add_argument('--width', type=int, default=320)
    parser.add_argument('--height', type=int, default=240)
    parser.add_argument('--camera-name', default='fixedfar')
    parser.add_argument('--mujoco-gl', default='egl')
    parser.add_argument('--out', default='figures/replays/ppo_reward_only_100k_seed0.gif')
    args = parser.parse_args()

    if args.mujoco_gl and args.mujoco_gl.lower() != 'none':
        os.environ.setdefault('MUJOCO_GL', args.mujoco_gl)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    first_frame_path = out_path.with_suffix('.png')

    model = PPO.load(args.model, device='cpu')
    env = safety_gymnasium.make(
        args.env,
        render_mode='rgb_array',
        camera_name=args.camera_name,
        width=args.width,
        height=args.height,
    )
    obs, info = env.reset(seed=args.seed)

    frames = []
    total_reward = 0.0
    total_cost = 0.0
    frames.append(annotate(env.render(), 0, 0.0, total_reward, 0.0, total_cost))

    for step in range(1, args.steps + 1):
        action, _state = model.predict(np.asarray(obs, dtype=np.float32), deterministic=True)
        obs, reward, cost, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        total_cost += float(cost)

        if step % args.frame_skip == 0 or terminated or truncated:
            frames.append(annotate(env.render(), step, float(reward), total_reward, float(cost), total_cost))

        if terminated or truncated:
            break

    env.close()
    imageio.imwrite(first_frame_path, frames[0])
    imageio.mimsave(out_path, frames, duration=0.08)

    print('frames:', len(frames))
    print('steps:', step)
    print('total_reward:', total_reward)
    print('total_cost:', total_cost)
    print('first_frame:', first_frame_path)
    print('replay:', out_path)


if __name__ == '__main__':
    main()
