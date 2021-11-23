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
