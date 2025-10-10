"""
Based on file from Elliot Tower from Pettingzoo (https://github.com/elliottower), which can be found here: https://pettingzoo.farama.org/tutorials/sb3/kaz/
"""
import glob
import os
import time
import multiprocessing
import argparse

import supersuit as ss
from stable_baselines3 import DQN, SAC, TD3, DDPG, A2C, PPO
from stable_baselines3.common.vec_env import VecMonitor

import particle_v1

def train(env_fn, steps: int = 1000000, seed: int | None = 0, render_mode: str | None = None, **env_kwargs):
    # Load environment and preprocess using supersuit
    env = env_fn.parallel_env(render_mode=render_mode, **env_kwargs)
    print(f"Starting training on {str(env.metadata['name'])} with {MODEL_TYPE}.")
    env.reset(seed=seed) 

    # Stack frames, so context of actions is included (helpful for learning movement)
    env = ss.frame_stack_v1(env, stack_size=4)

    # This is crucial for combining multi-agent observations
    env = ss.multiagent_wrappers.pad_observations_v0(env)

    # Convert parallel to Markov VectorEnv (needed for SB3). This treats each agent as a separate environment, so if num_agents=10, this is like having 10 envs.
    env = ss.pettingzoo_env_to_vec_env_v1(env)

    # Concatenate multiple environments for SB3 -> SB3VecEnvWrapper. We can now train on several vector environments in parallel (per agent).
    # num_vec_envs is per agent, so in total we train on num_vec_envs*num_agents environments --> should be a multiple of num_cpus for best performance.
    env = ss.concat_vec_envs_v1(vec_env=env, num_vec_envs=1, num_cpus=NUM_CORES, base_class="stable_baselines3") 

    # This helps tensorboard logging
    env = VecMonitor(env, filename=(LOG_PATH + '/' + MODEL_TYPE))

    # Initiate model from list = DQN, SAC, TD3, DDPG, A2C, PPO
    if MODEL_TYPE == 'DQN':
        model = DQN('MlpPolicy', env, verbose=3, tensorboard_log = LOG_PATH) #batch_size=256, )
    elif MODEL_TYPE == 'SAC':
        model = SAC('MlpPolicy', env, verbose=3, tensorboard_log = LOG_PATH)
    elif MODEL_TYPE == 'TD3':
        model = TD3('MlpPolicy', env, verbose=3, tensorboard_log = LOG_PATH)
    elif MODEL_TYPE == 'DDPG':
        model = DDPG('MlpPolicy', env, verbose=3, tensorboard_log = LOG_PATH)
    elif MODEL_TYPE == 'A2C':
        model = A2C('MlpPolicy', env, verbose=3, tensorboard_log = LOG_PATH)
    elif MODEL_TYPE == 'PPO':
        model = PPO('MlpPolicy', env, verbose=3, tensorboard_log = LOG_PATH)
    
    # Train and save model
    model.learn(total_timesteps=steps)
    model.save(f"models/{MODEL_TYPE}_{time.strftime('%Y%m%d-%H%M%S')}")
    print("Model has been saved.")
    print(f"Finished training on {str(env.unwrapped.metadata['name'])}.")

    env.close()


def eval(env_fn, num_games: int = 100, render_mode: str | None = None, **env_kwargs):
    # Evaluate a trained agent vs a random agent
    env = env_fn.env(render_mode = None, **env_kwargs)
    env = ss.frame_stack_v1(env, stack_size=4)

    print(
        f"\nStarting evaluation on {str(env.metadata['name'])} (num_games={num_games}, render_mode={render_mode})"
    )

    try:
        latest_policy = max(
            glob.glob(f"models/{MODEL_TYPE}*.zip"), key=os.path.getctime
        )
    except ValueError:
        print("Policy not found.")
        exit(0)

    # Load saved model
    if MODEL_TYPE == 'DQN':
        model = DQN.load(latest_policy)
    elif MODEL_TYPE == 'SAC':
        model = SAC.load(latest_policy)
    elif MODEL_TYPE == 'TD3':
        model = TD3.load(latest_policy)
    elif MODEL_TYPE == 'DDPG':
        model = DDPG.load(latest_policy)
    elif MODEL_TYPE == 'A2C':
        model = A2C.load(latest_policy)
    elif MODEL_TYPE == 'PPO':
        model = PPO.load(latest_policy)
    
    rewards = {agent: 0 for agent in env.possible_agents}

    # Note: we evaluate here using an AEC environments, to allow for easy A/B testing against random policies
    # For example, we can see here that using a random agent for archer_0 results in less points than the trained agent
    for i in range(num_games):
        env.reset(seed=i)
        env.action_space(env.possible_agents[0]).seed(i)

        for agent in env.agent_iter():
            obs, reward, termination, truncation, info = env.last()

            for a in env.agents:
                rewards[a] += env.rewards[a]

            if termination or truncation:
                break
            else:
                if agent == env.possible_agents[0]:
                    act = env.action_space(agent).sample()
                else:
                    act = model.predict(obs, deterministic=True)[0]
            env.step(act)
    env.close()

    avg_reward = sum(rewards.values()) / len(rewards.values())
    avg_reward_per_agent = {
        agent: rewards[agent] / num_games for agent in env.possible_agents
    }
    print(f"Avg reward: {avg_reward}")
    print("Avg reward per agent, per game: ", avg_reward_per_agent)
    print("Full rewards: ", rewards)
    return avg_reward


if __name__ == "__main__":
    # Needed on MacOS(according to supersuit doc)
    multiprocessing.set_start_method("fork")
    
    # Make tensorboard log dir
    os.makedirs("logs/", exist_ok=True)
    LOG_PATH = "logs/"

    # Get number of cpus
    NUM_CORES = multiprocessing.cpu_count()

    # Get env function
    env_fn = particle_v1
    
    # Set vector_state to false in order to use visual observations (significantly longer training time)
    env_kwargs = dict(num_agents=10, num_food_sources=1, flow='none', max_cycles=100)

    # Get model type through parser
    parser = argparse.ArgumentParser(description="Enter RL algorithm type.")
    parser.add_argument('model_type', help='Stable Baselines3 RL algorithm')
    args = parser.parse_args()

    if args.model_type in ['DQN', 'SAC', 'TD3', 'DDPG', 'A2C', 'PPO']:
        MODEL_TYPE = args.model_type
    else:
        print('No correct model type given. Options are: DQN, SAC, TD3, DDPG, A2C, PPO.')

    # Train a model (takes ~5 minutes on a laptop CPU)
    train(env_fn, steps=1000000, seed=0, render_mode = None, **env_kwargs)

    # Evaluate 10 games without rendering
    eval(env_fn, num_games=10, render_mode=None, **env_kwargs)

    # Watch 2 games in pygame window
    eval(env_fn, num_games=2, render_mode="human", **env_kwargs)