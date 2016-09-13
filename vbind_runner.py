import itertools
import os
import shutil
import subprocess

cells = ['gru',
         'lstm',
         'cp-gate-combined']
lengths = ['100', '200', '500']
nums = ['1', '2', '3']

grid_iter = itertools.product(cells, lengths, nums)

for cell, length, num_items in grid_iter:
    for i in range(5):
        results_dir = os.path.join(
            'cont_grid',
            '{}x{}'.format(length, num_items),
            'cell',
            '{}'.format(i))
        os.makedirs(results_dir)
        width = int(num_items)*10
        args = ['python',
                'vbind.py',
                '--width={}'.format(width),
                '--rank={}'.format(width),
                '--task=continuous',
                '--batch_size=32',
                '--num_steps=5000',
                '--learning_rate=0.01',
                '--cell={}'.format(cell),
                '--sequence_length={}'.format(length),
                '--num_items={}'.format(num_items),
                '--results_dir={}'.format(results_dir)]
        subprocess.run(args, check=True)
