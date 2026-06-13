import argparse
import numpy as np
import safety_gymnasium


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="SafetyPointGoal1-v0")
    args = parser.parse_args()

    env = safety_gymnasium.make(args.env)
    obs, info = env.reset(seed=0)

    print("env:", args.env)
    print("observation_space:", env.observation_space)
    print("action_space:", env.action_space)
    print("obs type:", type(obs))
    print("obs shape:", np.asarray(obs).shape)
    print("obs first values:", np.asarray(obs)[:20])
    print("obs last values:", np.asarray(obs)[-20:])

    if hasattr(env, "obs_space_dict"):
        print("\nobs_space_dict:")
        start = 0
        for key, space in env.obs_space_dict.items():
            dim = int(np.prod(space.shape))
            print(f"{key}: slice({start}, {start + dim}), shape={space.shape}")
            start += dim
    else:
        print("\nNo env.obs_space_dict found. Use tail masking as fallback.")

    env.close()


if __name__ == "__main__":
    main()