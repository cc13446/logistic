import json


def get_input_train():
    data = []
    label = []
    load_json = json.load(open('resource/trainset.json'))
    for j in load_json:
        data.append(j[0])
        label.append(j[1])

    return data, label


def get_input_dev():
    data = []
    label = []
    load_json = json.load(open('resource/devset.json'))
    for j in load_json:
        data.append(j[0])
        label.append(j[1])

    return data, label


def get_input_test():
    load_json = json.load(open('resource/testset.json'))
    return load_json


def output_test(data, label, path):
    output_json = []
    for i in range(0, len(data)):
        output_json.append([data[i], label[i]])
    with open(path, 'w') as f:
        json.dump(output_json, f)
