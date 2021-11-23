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
    # 训练数据长度
    _train_len = 0
    # 验证数据 len x t_len
    _valid_data = []
    # 验证答案 1 x len
    _valid_label = []
    # 验证数据 len x t_len
    _valid_data_mat = [[]]
    # 验证答案 1 x len
    _valid_label_mat = [[]]
    # 验证数据长度
    _valid_len = 0
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
    # 验证损失
    _valid_loss = []
    # 准确度
    _train_accuracy = 0
    _valid_accuracy = 0

    def get_dev_loss(self):
        return self._valid_loss

    def get_train_loss(self):
        return self._train_loss

    def print_train_accuracy(self):
        result = sigmoid((self._train_data_mat * self._T))
        res = 0
        for i in range(0, len(self._train_label)):
            if self._train_label[i] == 1 and result[i][0] >= 0.5:
                res += 1
            elif self._train_label[i] == 0 and result[i][0] < 0.5:
                res += 1
        print("Accuracy of train set: %.6f" % (1.0 * res / len(self._train_label)))

    def print_valid_accuracy(self):
        result = sigmoid((self._valid_data_mat * self._T))
        res = 0
        for i in range(0, len(self._valid_label)):
            if self._valid_label[i] == 1 and result[i][0] >= 0.5:
                res += 1
            elif self._valid_label[i] == 0 and result[i][0] < 0.5:
                res += 1
        print("Accuracy of valid set: %.6f" % (1.0 * res / len(self._valid_label)))

    def start_train(self):
        for k in range(0, self._max_cycles):
            self.loss_train()
            self.loss_valid()
            self.optimize()
            if k % 100 == 0:
                print("Cost after iteration %i---train: %f; valid: %f" % (k, self._train_loss[k], self._valid_loss[k]))
            if k > 3 and self._train_loss[k] == self._train_loss[k - 1] \
                    and self._train_loss[k] == self._train_loss[k - 2]:
                print("Converged after iteration %i---train: %f; valid: %f" % (k, self._train_loss[k], self._valid_loss[k]))
                break
        self.print_train_accuracy()
        self.print_valid_accuracy()

    def optimize(self):
        all_y = sigmoid(self._train_data_mat * self._T)
        self._T = self._T + self._learn_rate * self._train_data_mat.transpose() * (self._train_label_mat.transpose()
                                                                                   - all_y) / self._train_len

    def loss_train(self):
        all_y = sigmoid(self._train_data_mat * self._T)
        self._train_loss.append(-((self._train_label_mat * np.log(all_y) + (1 - self._train_label_mat)
                                   * np.log(1 - all_y)) / self._train_len).getA()[0][0])

    def loss_valid(self):
        all_y = sigmoid(self._valid_data_mat * self._T)
        self._valid_loss.append(-((self._valid_label_mat * np.log(all_y) + (1 - self._valid_label_mat)
                                   * np.log(1 - all_y)) / self._valid_len).getA()[0][0])

    def handle_train_data(self):
        for i in range(0, len(self._train_data)):
            self._train_data[i] = np.append(np.array([1.0]), self._train_data[i])

    def handle_valid_data(self):
        for i in range(0, len(self._valid_data)):
            self._valid_data[i] = np.append(np.array([1.0]), self._valid_data[i])

    def __init__(self, data, label, dev_data, dev_label, learn_rate, max_cycles):
        self._train_data = data
        self.handle_train_data()
        self._train_label = label
        self._train_data_mat = mat(self._train_data)
        self._train_label_mat = mat(self._train_label)

        self._valid_data = dev_data
        self.handle_valid_data()
        self._valid_label = dev_label
        self._valid_data_mat = mat(self._valid_data)
        self._valid_label_mat = mat(self._valid_label)

        self._learn_rate = learn_rate
        self._max_cycles = max_cycles

        self._train_len = len(self._train_data)
        self._valid_len = len(self._valid_data)

        self._t_len = data[0].shape[0]
        self._T = np.zeros([self._t_len, 1], dtype=float)
