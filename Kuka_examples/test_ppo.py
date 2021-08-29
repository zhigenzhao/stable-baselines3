import os
from stable_baselines3.common import callbacks
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.ppo.policies import MlpPolicy
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

from .kuka_planar_env import KukaPlanarEnv
from .save_on_best_training_reward_callback import SaveOnBestTrainingRewardCallback


def main():
    # env = gym.make("CartPole-v1")
    file_dir = os.path.dirname(os.path.abspath(__file__))
    log_dir = os.path.join(file_dir, "saved_models")
    try:
        os.mkdir(log_dir)
    except:
        pass

    env = KukaPlanarEnv(q_goal=np.array([1, -1, -1]), N=300, renders=True)
    env = Monitor(env, log_dir)

    callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=log_dir)

    model = PPO(MlpPolicy, env, verbose=1, batch_size=64,gae_lambda=0.98,gamma=0.999, ent_coef=0.0001, learning_rate=1e-4)
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100)

    print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")

    total_timesteps=int(2e7)
    model.learn(total_timesteps=total_timesteps, callback=callback)
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100)
    print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")

    model.save(log_dir+"/PPO_Kuka"+str(total_timesteps))

if __name__=="__main__":
    main()