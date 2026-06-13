import safety_gymnasium


def main():
    env = safety_gymnasium.make("SafetyPointGoal1-v0")
    obs, info = env.reset(seed=0)

    total_reward = 0.0
    total_cost = 0.0

    for _ in range(100):
        action = env.action_space.sample()
        obs, reward, cost, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        total_cost += float(cost)

        if terminated or truncated:
            obs, info = env.reset()

    env.close()
    print({
        "total_reward": total_reward,
        "total_cost": total_cost,
    })


if __name__ == "__main__":
    main()