import os
import os.path as osp

for file in os.listdir('./'):
    if 'rl' in file or 'nipa' in file or 'meta' in file:
        continue
    if osp.isfile(file):
        os.system('CUDA_VISIBLE_DEVICES=2 python %s' % file)


