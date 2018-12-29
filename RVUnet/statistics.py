import json, os

def read_json(path):
    try:
        json_read = open(path, 'r')
    except Exception as e:
        print("创建Json文件")
        json_write = open(path, 'wb')
        json_write.close()
        json_read = open(path, 'r')
    try:
        video_list = json.loads(json_read.read())
    except:
        video_list = []
    json_read.close()
    return video_list


# 将字典写入json文件
def write_json(video_list, path):
    try:
        json_write = open(path, 'w')
        json.dump(video_list, json_write)
        json_write.close()
    except Exception as e:
        print(e)
        return False
    return True


def init_statistics(save_path, **kwargs):
    write_json(kwargs, os.path.join(save_path, 'statistics.json'))
    statistics = read_json(os.path.join(save_path, 'statistics.json'))
    statistics['result'] = []
    statistics['result'].append({'train': {},
                                 'val': {},
                                 'train_avg': [0, 0, 0],
                                 'val_avg': [0, 0, 0],
                                 'train_add': [0, 0, 0],
                                 'val_add': [0, 0, 0]
                                 })
    statistics['train_best'] = [0,0,0]
    statistics['val_best'] = [0,0,0]
    write_json(statistics, os.path.join(save_path, 'statistics.json'))


def add_train(epoch_num, save_path, volume_name, ious ,over):
    statistics = read_json(os.path.join(save_path, 'statistics.json'))
    try:
        statistics['result'][epoch_num]['train'][volume_name] = ious
        train_avg = statistics['result'][epoch_num]['train_avg']
        length = len(list(statistics['result'][epoch_num]['train'].values()))
        train_avg = [(train_avg[0] * (length - 1) + ious[0]) / length, (train_avg[1] * (length - 1) + ious[1]) / length,
                     (train_avg[2] * (length - 1) + ious[2]) / length]
        statistics['result'][epoch_num]['train_avg'] = train_avg

        if over:
            last_train_avg = statistics['result'][epoch_num - 1]['train_avg']
            statistics['result'][epoch_num]['train_add'] = [a - b for a, b in zip(train_avg, last_train_avg)]
            statistics['train_best'] = [a if a>b else b for a,b in zip(statistics['train_best'],train_avg)]


    except:
        statistics['result'].append({'train': {},
                                     'val': {},
                                     'train_avg': [0, 0, 0],
                                     'val_avg': [0, 0, 0],
                                     'train_add': [0, 0, 0],
                                     'val_add': [0, 0, 0]
                                     })
        statistics['result'][epoch_num]['train'][volume_name] = ious
        train_avg = statistics['result'][epoch_num]['train_avg']
        length = len(list(statistics['result'][epoch_num]['train'].values()))
        train_avg = [(train_avg[0] * (length - 1) + ious[0]) / length, (train_avg[1] * (length - 1) + ious[1]) / length,
                     (train_avg[2] * (length - 1) + ious[2]) / length]

        statistics['result'][epoch_num]['train_avg'] = train_avg

        if over:
            last_train_avg = statistics['result'][epoch_num - 1]['train_avg']
            statistics['result'][epoch_num]['train_add'] = [a - b for a, b in zip(train_avg, last_train_avg)]
            statistics['train_best'] = [a if a>b else b for a,b in zip(statistics['train_best'],train_avg)]

    write_json(statistics, os.path.join(save_path, 'statistics.json'))


def add_val(epoch_num, save_path, volume_name, ious,over):
    statistics = read_json(os.path.join(save_path, 'statistics.json'))
    try:
        statistics['result'][epoch_num]['val'][volume_name] = ious
        val_avg = statistics['result'][epoch_num]['val_avg']
        length = len(list(statistics['result'][epoch_num]['val'].values()))
        val_avg = [(val_avg[0] * (length - 1) + ious[0]) / length, (val_avg[1] * (length - 1) + ious[1]) / length,
                   (val_avg[2] * (length - 1) + ious[2]) / length]
        statistics['result'][epoch_num]['val_avg'] = val_avg

        if over:
            last_val_avg = statistics['result'][epoch_num - 1]['val_avg']
            statistics['result'][epoch_num]['val_add'] = [a - b for a, b in zip(val_avg, last_val_avg)]
            statistics['val_best'] = [a if a>b else b for a,b in zip(statistics['val_best'],val_avg)]

    except:
        statistics['result'].append({'train': {},
                                     'val': {},
                                     'train_avg': [0, 0, 0],
                                     'val_avg': [0, 0, 0]
                                     })
        statistics['result'][epoch_num]['val'][volume_name] = ious
        val_avg = statistics['result'][epoch_num]['val_avg']
        length = len(list(statistics['result'][epoch_num]['val'].values()))
        val_avg = [(val_avg[0] * (length - 1) + ious[0]) / length, (val_avg[1] * (length - 1) + ious[1]) / length,
                   (val_avg[2] * (length - 1) + ious[2]) / length]
        statistics['result'][epoch_num]['val_avg'] = val_avg

        if over:
            last_val_avg = statistics['result'][epoch_num - 1]['val_avg']
            statistics['result'][epoch_num]['val_add'] = [a - b for a, b in zip(val_avg, last_val_avg)]
            statistics['val_best'] = [a if a>b else b for a,b in zip(statistics['val_best'],val_avg)]

    write_json(statistics, os.path.join(save_path, 'statistics.json'))
