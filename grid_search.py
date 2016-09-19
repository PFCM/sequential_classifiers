"""Runs a bunch of them"""
import itertools
import subprocess
import time
import shutil
import logging
import sys
import os

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.FileHandler('log')
handler.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(message)s')
handler.setFormatter(formatter)

logger.addHandler(handler)

args = [
    'python',
    os.path.abspath(
        os.path.join(
            os.environ['HOME'],
            'COMP489',
            'sequential_classifiers',
            'mnist.py')),
    '--width=100',
    '--layers=1',
    '--num_epochs=100',
    '--batch_size=100',
    '--permute=True',
    '--optimiser=adam',
]

cells = [
    'cp-gate-combined',
    'cp-gate-combined-linear',
    'cp-gate-linear'
#    'cp-loss',
#    'cp-del',
#    'cp+-',
#    'cp+',
#    'lstm',
    'vanilla',
#    'irnn'
]

learning_rates = ['0.01']
grad_norms = ['1000.0']
ranks = ['50', '100']
#ranks = ['150', '200']
#ranks = ['100']

twidth = shutil.get_terminal_size((80, 20)).columns

def run_subprocess(args):
    logger.info('starting run: %s', '\n'.join(args))
    start = time.time()
    subprocess.run(args, check=True)
    end = time.time()
    logger.info('run finished in %f s', end-start)
    

if len(sys.argv) == 1:  # full sequential search

    for cell, rank in itertools.product(cells, ranks):
        if cell in ['lstm', 'vanilla', 'irnn'] \
           and rank != ranks[0]:
            continue
        print('{:/^{}}'.format('{}-{}'.format(cell, rank), twidth))
        unique_args = [
            '--cell='+cell,
            '--rank='+rank,
            '--results_dir=mnist_perms-lineargates/{}-{}'.format(cell, rank),
            '--learning_rate=0.01',
            '--max_grad_norm=1.0']
        run_subprocess(args + unique_args)
elif len(sys.argv) == 2:
    # then we are just doing one and the grid has told us which
    cell, rank, lr, gn = list(itertools.product(cells, ranks, learning_rates, grad_norms))[int(sys.argv[1]) - 1]
    unique_args = [
        '--cell='+cell,
        '--rank='+rank,
        '--results_dir=output',
        '--max_grad_norm='+gn,
        '--learning_rate='+lr]
    with open('model_details.txt', 'w') as fp:
        fp.write('{}, rank {}, lr {}, grad clip {}\n'.format(cell, rank, lr, gn))
    print('Doing {}, rank {}'.format(cell, rank))
    run_subprocess(args + unique_args)
else:
    print(sys.argv, 'No idea.')
