import os
import gym
import numpy as np
from numpy.lib.npyio import save
from stable_baselines3 import PPO
from stable_baselines3.ppo.policies import MlpPolicy
from stable_baselines3.common.evaluation import evaluate_policy

from .kuka_planar_env import KukaPlanarEnv

def main():
    # env = gym.make("CartPole-v1")
    env = KukaPlanarEnv(q_goal=np.array([1, -1, -1]), N=300, renders=False)
    model = PPO(MlpPolicy, env, verbose=1)
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100)

    print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")

    total_timesteps=1e7
    model.learn(total_timesteps=total_timesteps)
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100)
    print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")
    file_dir = os.path.dirname(os.path.abspath(__file__))
    save_dir = os.path.join(file_dir, "saved_models")
    try:
        os.mkdir(save_dir)
    except:
        pass
    model.save(save_dir+"/PPO_Kuka"+str(total_timesteps))

if __name__=="__main__":
    main()