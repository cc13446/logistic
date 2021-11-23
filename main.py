import dataHelper
import logistic


def main():
    data, label = dataHelper.get_input_train()
    model = logistic.Logistic(data, label, 0.1, 50000)
    model.start_train()

    data, label = dataHelper.get_input_dev()
    model.dev_test(data, label)


if __name__ == "__main__":
    main()
