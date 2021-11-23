import dataHelper
import logistic


def main():
    # 训练
    data, label = dataHelper.get_input_train()
    valid_data, valid_label = dataHelper.get_input_dev()
    model = logistic.Logistic(data, label, valid_data, valid_label, 0.1, 50000)
    model.start_train()


if __name__ == "__main__":
    main()
