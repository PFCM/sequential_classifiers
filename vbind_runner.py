import itertools
import os
import shutil
import subprocess

cells = ['tguv2']

# lengths = ['100', '200', '500']
lengths = ['100']
nums = ['1', '2', '3']
# lrates = ['0.01', '0.001', '0.0001']
lrates = ['0.01', '0.1']

grid_iter = itertools.product(cells, lengths, nums, lrates)

for cell, length, num_items, lr in grid_iter:
    for i in range(5):
        results_dir = os.path.join(
            '../2017/variable_binding/longruns/noisy/tanh',
            '{}x{}'.format(length, num_items),
            cell,
            lr,
            '{}'.format(i))
        os.makedirs(results_dir)
        width = int(num_items)*10
        args = ['python',
                'vbind.py',
                '--width={}'.format(width),
                '--rank={}'.format(width),
                '--task=continuous',
                '--inbetween_noise=True',
                '--batch_size=50',
                '--num_steps=25000',
                '--learning_rate={}'.format(lr),
                '--cell={}'.format(cell),
                '--sequence_length={}'.format(length),
                '--num_items={}'.format(num_items),
                '--results_dir={}'.format(results_dir)]
        subprocess.run(args, check=True)
