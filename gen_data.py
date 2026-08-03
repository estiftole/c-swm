"""Simple random agent.

Running this script directly executes the random agent in environment and stores
experience in a replay buffer.
"""

import argparse
import sys
import os
from pathlib import Path


if str(Path.cwd()) not in sys.path:
    sys.path.insert(0, str(Path.cwd()))

import numpy as np
from PIL import Image

# Use modern gymnasium (or modern gym)
import gymnasium as gym
import ale_py  # <-- ADD THIS LINE! This registers ALE/Pong-v5, etc.

# Uncomment if using custom C-SWM local environments (e.g. ShapesTrain-v0)
# import envs
import utils


class RandomAgent(object):
    """The world's simplest agent!"""

    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, observation, reward, done):
        del observation, reward, done
        return self.action_space.sample()


def crop_normalize(img, crop_ratio):
    img = img[crop_ratio[0]:crop_ratio[1]]
    img = Image.fromarray(img).resize((50, 50), Image.Resampling.LANCZOS)
    return np.transpose(np.array(img), (2, 0, 1)) / 255.0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--env_id', type=str, default='ShapesTrain-v0',
                        help='Select the environment to run.')
    parser.add_argument('--fname', type=str, default='data/shapes_train.h5',
                        help='Save path for replay buffer.')
    parser.add_argument('--num_episodes', type=int, default=1000,
                        help='Total number of episodes to simulate.')
    parser.add_argument('--atari', action='store_true', default=False,
                        help='Run atari mode (stack multiple frames).')
    parser.add_argument('--seed', type=int, default=1,
                        help='Random seed.')
    args = parser.parse_args()

    crop = None
    warmstart = None
    env_id_lower = args.env_id.lower()

    if 'pong' in env_id_lower:
        crop = (35, 190)
        warmstart = 58
    elif 'spaceinvaders' in env_id_lower:
        crop = (30, 200)
        warmstart = 50
    elif args.atari:
        # Default fallback for other Atari games if not explicitly matched above
        crop = (0, 210)
        warmstart = 0

    # Ensure warmstart is an integer if atari mode is enabled
    if args.atari:
        warmstart = warmstart if warmstart is not None else 0

    # Pass max_episode_steps directly to gym.make if running atari mode
    kwargs = {}
    if args.atari and warmstart is not None:
        kwargs['max_episode_steps'] = warmstart + 11

    # Create Environment
    env = gym.make(args.env_id, **kwargs)

    np.random.seed(args.seed)
    env.action_space.seed(args.seed)

    agent = RandomAgent(env.action_space)

    episode_count = args.num_episodes
    reward = 0
    done = False

    replay_buffer = []

    for i in range(episode_count):

        replay_buffer.append({
            'obs': [],
            'action': [],
            'next_obs': [],
        })

        # Modern reset returns (observation, info)
        # We pass episode-specific seed for reproducible rollouts
        ep_seed = args.seed + i
        ob, info = env.reset(seed=ep_seed)

        if args.atari:
            # Burn-in steps
            for _ in range(warmstart):
                action = agent.act(ob, reward, done)
                # Modern step returns 5 values: ob, reward, terminated, truncated, info
                ob, _, _, _, _ = env.step(action)

            prev_ob = crop_normalize(ob, crop)
            ob, _, _, _, _ = env.step(0)
            ob = crop_normalize(ob, crop)

            while True:
                replay_buffer[i]['obs'].append(
                    np.concatenate((ob, prev_ob), axis=0))
                prev_ob = ob

                action = agent.act(ob, reward, done)
                ob, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                ob = crop_normalize(ob, crop)

                replay_buffer[i]['action'].append(action)
                replay_buffer[i]['next_obs'].append(
                    np.concatenate((ob, prev_ob), axis=0))

                if done:
                    break
        else:
            while True:
                # Custom envs like ShapesTrain-v0 return ob as a tuple (state, image_array)
                obs_frame = ob[1] if isinstance(ob, (tuple, list)) else ob
                replay_buffer[i]['obs'].append(obs_frame)

                action = agent.act(ob, reward, done)
                ob, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                next_obs_frame = ob[1] if isinstance(ob, (tuple, list)) else ob
                replay_buffer[i]['action'].append(action)
                replay_buffer[i]['next_obs'].append(next_obs_frame)

                if done:
                    break

        if i % 10 == 0:
            print(f"iter {i}")

    env.close()

    # Save replay buffer to disk.
    utils.save_list_dict_h5py(replay_buffer, args.fname)
# uv run gen_data.py --env_id ALE/Pong-v5 --fname data/pong_train.h5 --num_episodes 1000 --atari --seed 1
# uv run gen_data.py --env_id ALE/Pong-v5 --fname data/pong_eval.h5 --num_episodes 100 --atari --seed 2
