from statistics import *

import re,random
time = 10
root = '/workspace/mnt/group/alg-pro/yankai/segment/data/pre_process'

file_list = [file for file in os.listdir(root) if int(re.sub("\D", "", file)) <= 130 and "volume" in file]
file_test_list = [file for file in os.listdir(root) if int(re.sub("\D", "", file)) > 130 and "volume" in file]


init_statistics("test",batch_size = 10)
for i in range(time):
    for j,file in enumerate(file_list):
        over  = (j == len(file_list)-1)

        add_train(i+1,"test",file,[random.randint(0,100)/100,random.randint(0,100)/100,random.randint(0,100)/100],over)

    for j,file in enumerate(file_test_list):
        over = (j == len(file_test_list) - 1)
        add_val(i+1,"test",file,[random.randint(0,100)/100,random.randint(0,100)/100,random.randint(0,100)/100],over)
