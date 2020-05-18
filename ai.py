import argparse

import gym

import agents
import models


def make_cli_parser():
    parser = argparse.ArgumentParser(description="Do AI things")
    parser.add_argument(
        "env", help="gym.Env to use", choices=["CartPole-v0", "FrozenLake-v0"]
    )
    parser.add_argument("agent", help="Agent to use", choices=["Random", "QLearning"])
    parser.add_argument(
        "--use-policy",
        action="store_true",
        help="Act according to model, no exploration",
    )
    parser.add_argument(
        "--no-save", action="store_true", help="Do not save any changes to the agent"
    )
    parser.add_argument(
        "--episodes",
        "-e",
        type=int,
        default=100,
        help="Episodes to play (default: 100)",
    )
    return parser


def loop(env: gym.Env, agent: agents.Agent, episodes: int, use_policy: bool):
    observation = env.reset()
    wins = 0
    episode = 1

    env.render()
    while episode <= episodes:
        if use_policy:
            action = agent.act(observation)
        else:
            action = agent.explore(observation)

        new_observation, reward, done, info = env.step(action)

        if not use_policy:
            agent.learn(observation, action, new_observation, reward)

        env.render()
        print("\tEpisodes:", episode, "\tWins:", wins)
        print("\tWin Ratio:", wins / episode)

        if done:
            observation = env.reset()
            episode += 1
            if reward:
                wins += 1
        else:
            observation = new_observation

    env.close()


def main():
    parser = make_cli_parser()
    args = parser.parse_args()

    env = gym.make(args.env)
    agent = models.load_model(args.env, args.agent)
    if not agent:
        agent = getattr(agents, args.agent)(env.action_space)

    loop(env, agent, args.episodes, args.use_policy)

    print(agent)

    if not args.no_save:
        models.save_model(args.env, args.agent, agent)


if __name__ == "__main__":
    main()
