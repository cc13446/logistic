import matplotlib.pyplot as plt
import numpy as np


def loss_picture_output(loss_train, loss_valid, path):
    x = np.arange(0, len(loss_train), 1)
    plt.plot(x, loss_train, label='train')
    plt.plot(x, loss_valid, label='valid')
    plt.xticks(np.arange(0, len(loss_train), 1000))
    plt.xticks(rotation=60)
    plt.legend()
    plt.savefig(path)
    plt.show()
