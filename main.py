import dataHelper
import logistic
import pictureHelper


def main():
    # 训练
    data, label = dataHelper.get_input_train()
    valid_data, valid_label = dataHelper.get_input_dev()
    model = logistic.Logistic(data, label, valid_data, valid_label, 0.1, 50000)
    model.start_train()

    # 画图
    pictureHelper.loss_picture_output(model.get_train_loss(), model.get_dev_loss(), "picture.png")

    # 测试
    test_data = dataHelper.get_input_test()
    test_label = model.get_test_label(test_data)
    dataHelper.output_test(test_data, test_label, 'testset.json')


if __name__ == "__main__":
    main()
