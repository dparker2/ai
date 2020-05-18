import argparse
from random import random

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
        "--epsilon",
        "-e",
        type=float,
        default=0.05,
        help="Chance of a random action being chosen (default: 0.05)",
    )
    parser.add_argument(
        "--num",
        "-n",
        type=int,
        default=100,
        help="Number of episodes to play (default: 100)",
    )
    parser.add_argument(
        "--no-learn",
        dest="learn",
        action="store_false",
        help="Do not let the agent learn",
    )
    parser.add_argument(
        "--no-save",
        dest="save",
        action="store_false",
        help="Do not save any changes to the agent",
    )
    return parser


def loop(
    env: gym.Env,
    agent: agents.Agent,
    num_episodes: int,
    epsilon: float,
    should_learn: bool,
):
    wins = 0
    episode = 1
    observation = env.reset()

    env.render()
    while episode <= num_episodes:
        if random() < epsilon:
            action = env.action_space.sample()
        else:
            action = agent.act(observation)

        new_observation, reward, done, info = env.step(action)

        if should_learn:
            agent.learn(observation, action, new_observation, reward)

        env.render()
        print("\tEpisodes:", episode, "\tWins:", wins)
        print("\tWin Ratio:", wins / episode)

        if done:
            observation = env.reset()
            env.render()
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

    loop(env, agent, args.num, args.epsilon, args.learn)

    print()
    print("Agent:\n", agent)

    if args.save:
        models.save_model(args.env, args.agent, agent)


if __name__ == "__main__":
    main()
