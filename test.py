import os,re
root = "/workspace/mnt/group/alg-pro/yankai/segment/data/pre_process"

file_list = [file for file in os.listdir(root) if int(re.sub("\D", "", file)) > 170]

for file in file_list:
    os.remove(os.path.join(root,file))