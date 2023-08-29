import os
import numpy as np

total_stock = 4565
per_stock = 400
N = int(np.ceil(total_stock / per_stock))
gpus = [0, 1]
cmd = ''
for n in range(N):
    cmd += f'CUDA_VISIBLE_DEVICES={gpus[n//12]} nohup python backtrader_sequence_model.py -n {n} --per_stock {per_stock} > log_v{n+1}.txt & '
os.system(cmd)