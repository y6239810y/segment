from model import LstmSegNet
from get_data import *
import re
from tools import *
from train import net_train
from val import net_val
import argparse
from statistics import *


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


parser = argparse.ArgumentParser(
    description='Liver segmentation with lstmConvNet by tensorflow')

parser.add_argument('--dataset_root', default="/workspace/mnt/group/alg-pro/yankai/segment/data/pre_process",
                    help='Dataset root directory path')

parser.add_argument('--save_path', default="test",
                    help='the path tensorboard and model save at')

parser.add_argument('--batch_size', default=12, type=int,
                    help='Batch size for training')

parser.add_argument('--resume', default=True, type=str2bool,
                    help='Checkpoint state_dict file to resume training from')

parser.add_argument('--cuda', default=True, type=str2bool,
                    help='Use CUDA to train model')

parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    help='initial learning rate')

parser.add_argument('--decay_rate', default=0.99, type=float,
                    help='rate of weight decay for learning_rate')

parser.add_argument('--decay_steps', default=300, type=int,
                    help='Weight decay steps for SGD')

parser.add_argument('--threshold', default=0.5, type=float,
                    help='threshold to filter probability map of liver')

parser.add_argument('--width', default=512, type=int,
                    help='the width of input data')

parser.add_argument('--height', default=512, type=int,
                    help='the height of input data')

parser.add_argument('--epochs', default=50, type=int,
                    help='the number of epochs for training')

parser.add_argument('--loss_func', default='cross_entropy', type=str,
                    help='loss_func for training')

parser.add_argument('--weights', default=[100, 10], type=list,
                    help='the weights of proportion between object and background')

parser.add_argument('--filter_no_liver', default=10, type=int,
                    help='the probability to filter on_object')

parser.add_argument('--down_sample', default=1, type=int,
                    help='the probability to filter on_object')

parser.add_argument('--norm_type', default='batch', type=str,
                    help='the type of normlization')

parser.add_argument('--supervise', default=1, type=str2bool,
                    help='the type of normlization')

parser.add_argument('--lstm', default=1, type=str2bool,
                    help='with lstm')

args = parser.parse_args()

args.width = args.width // args.down_sample

args.height = args.height // args.down_sample

if __name__ == '__main__':

    layers = [
        'CONV_1', 'RV_1', 'POOL_1', 'CONV_2', 'RV_2', 'POOL_2', 'CONV_3', 'RV_3', 'POOL_3',
        'CONV_4','CONV_5',
        'UPSAMPLE_1', 'CONBINE_1', 'RV_4', 'CONV_6', 'UPSAMPLE_2', 'CONBINE_2', 'RV_5', 'CONV_7',
        'UPSAMPLE_3', 'CONBINE_3', 'RV_6', 'CONV_LAST',
    ]
    layers_kernels = [
        {"kernel": [3, 3], "stride": 1, "filter": 64, "norm": args.norm_type}, # CONV_1

        {"filter":64,"norm": args.norm_type},  # RV_1

        {'pool_way': Max_Pooling_2d, "kernel": [2, 2], "stride": 2},  # POOL_1

        {"kernel": [3, 3], "stride": 1, "filter": 128, "norm": args.norm_type},  # CONV_2

        {"filter": 128, "norm": args.norm_type},  # RV_2

        {'pool_way': Max_Pooling_2d, "kernel": [2, 2], "stride": 2},  # POOL_2

        {"kernel": [3, 3], "stride": 1, "filter": 256, "norm": args.norm_type},  # CONV_3

        {"filter": 256, "norm": args.norm_type},  # RV_3

        {'pool_way': Max_Pooling_2d, "kernel": [2, 2], "stride": 2},  # POOL_3

        {"kernel": [3, 3], "stride": 1, "filter": 512, "norm": args.norm_type},  # CONV_1

        {"kernel": [3, 3], "stride": 1, "filter": 256, "norm": args.norm_type},  # CONV_1


        {"kernel": [3, 3], "stride": [2, 2], "filter": 128},  # UPSAMPLE_1
        {"add_layer": 'RV_3', "kernel": [1, 1]},  # CONBINE_1
        {"filter": 256, "norm": args.norm_type},  # RV_4
        {"kernel": [3, 3], "stride": 1, "filter": 256, "norm": args.norm_type},  # CONV_6

        {"kernel": [3, 3], "stride": [2, 2], "filter": 64},  # UPSAMPLE_2
        {"add_layer": 'RV_2', "kernel": [1, 1]},  # CONBINE_2
        {"filter": 128, "norm": args.norm_type},  # RV_5
        {"kernel": [3, 3], "stride": 1, "filter": 128, "norm": args.norm_type},  # CONV_7

        {"kernel": [3, 3], "stride": [2, 2], "filter": 32},  # UPSAMPLE_3
        {"add_layer": 'RV_1', "kernel": [1, 1]},  # CONBINE_3
        {"filter": 64, "norm": args.norm_type},  # RV_6
        {"kernel": [3, 3], "stride": 1, "filter": 2, "norm": args.norm_type},  # CONV_LAST

    ]

    model = LstmSegNet(layers=layers, layers_kernels=layers_kernels,
                       threshold=args.threshold, save_path=args.save_path, learning_rate=args.lr,
                       decay_steps=args.decay_steps, decay_rate=args.decay_rate, batch_size=args.batch_size,
                       width=args.width, height=args.height, resume=args.resume, loss_func=args.loss_func
                       )
    file_list = [file for file in os.listdir(args.dataset_root) if
                 int(re.sub("\D", "", file)) <= 110 and "volume" in file]
    file_test_list = [file for file in os.listdir(args.dataset_root) if
                      int(re.sub("\D", "", file)) > 110 and int(re.sub("\D", "", file)) <= 130 and "volume" in file]

    avg_liver_best = [0 for i in range(len(file_test_list))]

    model.train_times = 0
    model.test_times = 0

    weights = args.weights
    init_statistics(save_path=args.save_path, batch_size=args.batch_size, learning_rate=args.lr,
                    decay_rate=args.decay_rate,
                    decay_steps=args.decay_steps, threshold=args.threshold, width=args.width, height=args.height,
                    loss_func=args.loss_func, weights=args.weights, filter_no_liver=args.filter_no_liver
                    )
    count = 5
    for time in range(args.epochs):
        net_train(model=model, root=args.dataset_root, weights=weights, times=time + 1, down_sample=args.down_sample)
        avg_liver = net_val(model=model, root=args.dataset_root, weights=weights, times=time + 1,
                            down_sample=args.down_sample)

        if (np.sum(avg_liver) > np.sum(avg_liver_best)):
            avg_liver_best = avg_liver
            model._store(True)
        else:
            model._store(False)
        print("avg_liver_best: " + str(np.sum(avg_liver_best) / len(file_test_list)))

        if weights[1] + 5 < 100:
            weights[1] += 5
        else:
            weights[1] = 100
