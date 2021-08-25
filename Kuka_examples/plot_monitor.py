from matplotlib import pyplot as plt
import numpy as np
import csv


def main():
    filename = "/home/zhigen/code/stable-baselines3/Kuka_examples/saved_models/monitor.csv"
    with open(filename) as csvfile:
        training_monitor = csv.reader(csvfile)
        
        reward = []
        for (i, row) in enumerate(training_monitor):
            if i > 1:
                reward.append(float(row[0]))
    
        plt.plot(np.arange(len(reward)), np.log10(-np.array(reward)))
        plt.show()

if __name__=="__main__":
    main()