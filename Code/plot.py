import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os

directory = './TrainLog/'

for filename in os.listdir(directory):
    print(filename)

    colors = ["red", "blue", "green"]

    name = filename

    data = np.loadtxt("./TrainLog/" + name)

    # print(data)

    sns.tsplot(data=data, color=colors[1], condition=name)

    plt.xlabel('Iter')
    plt.ylabel('ThetaCenterReward')
    plt.tight_layout()
    plt.savefig("./TrainFig/" + name + ".png")
    plt.show()  # show the plot
    # break
