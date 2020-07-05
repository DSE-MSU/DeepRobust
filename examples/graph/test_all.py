import os
import os.path as osp

for file in os.listdir('./'):
    if "py" not in file:
        continue
    if 'rl' in file or 'nipa' in file or 'meta' in file or 'all' in file:
        continue
    if osp.isfile(file):
        print(file)
        os.system('CUDA_VISIBLE_DEVICES=0 python %s' % file)


