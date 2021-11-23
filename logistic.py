import numpy as np
from numpy import mat


def sigmoid(y):
    return 1.0 / (1 + np.exp(-y))


class Logistic:
    # 训练数据 len x t_len
    _train_data = []
    # 训练答案 1 x len
    _train_label = []
    # 训练数据 len x t_len
    _train_data_mat = [[]]
    # 训练答案 1 x len
    _train_label_mat = [[]]
    # 数据长度
    _len = 0
    # 参数长度
    _t_len = 1
    # 参数数组 t_len X 1
    _T = np.zeros([_t_len, 1], dtype=float)
    # 学习率
    _learn_rate = 0.0
    # 最大迭代次数
    _max_cycles = 500
    # 训练损失
    _train_loss = []
    # 测试准确度
    _test_accuracy = 0

    def dev_test(self, data, label):
        self._test_accuracy = 0
        for i in range(0, len(data)):
            data[i] = np.append(np.array([1.0]), data[i])
        data = mat(data)
        result = sigmoid((data * self._T))
        res = 0
        for i in range(0, len(label)):
            if label[i] == 1 and result[i][0] >= 0.5:
                res += 1
            elif label[i] == 0 and result[i][0] < 0.5:
                res += 1
        print("Accuracy of dev set: %.6f" % (1.0 * res / len(label)))

    def start_train(self):
        for k in range(0, self._max_cycles):
            self.loss()
            self.optimize()
            if k % 100 == 0:
                print("Cost after iteration %i: %f" % (k, self._train_loss[k]))
            if k > 3 and self._train_loss[k] == self._train_loss[k - 1] and self._train_loss[k] == self._train_loss[
                k - 2]:
                print("Converged after iteration %i: %f" % (k, self._train_loss[k]))
                break

    def optimize(self):
        all_y = sigmoid(self._train_data_mat * self._T)
        self._T = self._T + self._learn_rate * self._train_data_mat.transpose() * (
                self._train_label_mat.transpose() - all_y) / self._len

    def loss(self):
        all_y = sigmoid(self._train_data_mat * self._T)
        self._train_loss.append(-((self._train_label_mat * np.log(all_y) + (1 - self._train_label_mat) * np.log(
            1 - all_y)) / self._len).getA()[0][0])

    def handle_data(self):
        for i in range(0, len(self._train_data)):
            self._train_data[i] = np.append(np.array([1.0]), self._train_data[i])

    def __init__(self, data, label, learn_rate, max_cycles):
        self._train_data = data
        self._train_label = label
        self._learn_rate = learn_rate
        self._max_cycles = max_cycles
        self.handle_data()
        self._train_data_mat = mat(self._train_data)
        self._train_label_mat = mat(self._train_label)
        self._t_len = data[0].shape[0]
        self._len = len(data)
        self._T = np.zeros([self._t_len, 1], dtype=float)
