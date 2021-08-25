import os
import numpy as np
from .kuka_planar_env import KukaPlanarEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import PPO

def main():
    model_filename = "/home/zhigen/code/stable-baselines3/Kuka_examples/saved_models/best_model.zip"
    model = PPO.load(model_filename)

    env = KukaPlanarEnv(q_goal=np.array([1, -1, -1]), N=300, renders=True)

    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100)
    print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")

    # total_timesteps=1e7
    # model.set_env(env)
    # model.learn(total_timesteps=total_timesteps)

    # file_dir = os.path.dirname(os.path.abspath(__file__))
    # save_dir = os.path.join(file_dir, "saved_models")
    # try:
    #     os.mkdir(save_dir)
    # except:
    #     pass
    # model.save(save_dir+"/PPO_Kuka")


if __name__=="__main__":
    main()