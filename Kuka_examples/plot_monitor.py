from matplotlib import pyplot as plt
import numpy as np
import csv


def main():
    filename = "/home/zhigen/code/stable-baselines3/Kuka_examples/saved_models/monitor (copy).csv"
    with open(filename) as csvfile:
        training_monitor = csv.reader(csvfile)
        
        reward = []
        for (i, row) in enumerate(training_monitor):
            if i > 1:
                reward.append(float(row[0]))
        reward = np.array(reward)

        reward_vec = []
        reward_std = []
        i = 0
        n = 5000
        while i < len(reward):
            if i+n < len(reward):
                temp = reward[i:i+n]
            else:
                temp = reward[i:]
            
            m = np.mean(temp)
            s = np.std(temp)

            reward_vec.append(m)
            reward_std.append(s)

            i += n


        plt.plot(np.arange(len(reward_vec))*n, -np.array(reward_vec), linewidth=3)
        plt.plot(np.arange(len(reward_vec))*n, np.ones_like(reward_vec)*1750, linestyle="dashed", linewidth=3)
        plt.fill_between(np.arange(len(reward_vec))*n, 
            -np.array(reward_vec)+np.array(reward_std),
            -np.array(reward_vec)-np.array(reward_std),
            alpha=0.5)
        plt.yscale("log")
        plt.xlabel("Timestep", fontsize=24)
        plt.xticks(fontsize=14)
        plt.ylabel("Cost", fontsize=24)
        plt.yticks(fontsize=16)
        plt.legend(["PPO", "VERONICA baseline"], fontsize=16)
        plt.xlim([0, 300000])
        plt.show()

if __name__=="__main__":
    main()