"""Runs a bunch of them"""
import itertools
import subprocess
import time
import shutil
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.FileHandler('prelims/log')
handler.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(message)s')
handler.setFormatter(formatter)

logger.addHandler(handler)

args = [
    'python',
    'mnist.py',
    '--width=100',
    '--layers=1',
    '--num_epochs=100',
    '--batch_size=50',
    '--learning_rate=0.01'
]

cells = [
    'lstm',
    'vanilla',
    'irnn',
    'cp-relu',
    'cp-tanh',
    'tt-relu',
    'tt-tanh'
]

ranks = ['10', '50']

twidth = shutil.get_terminal_size((80, 20)).columns

for cell, rank in itertools.product(cells, ranks):
    if cell in ['lstm', 'vanilla', 'irnn'] \
       and rank != ranks[0]:
        break
    unique_args = [
        '--cell='+cell,
        '--rank='+rank,
        '--results_dir=prelims/{}-{}'.format(cell, rank)]

    print('~' * twidth)
    print('~' * twidth)
    print('{:~^{}}'.format('  '+cell+'  ', twidth))
    print('{:~^{}}'.format('  {}  '.format(rank), twidth))
    logger.info('starting run: %s-%s', cell, rank)
    start = time.time()
    subprocess.run(args + unique_args, check=True)
    end = time.time()
    logger.info('run finished in %f s', end-start)
    print('{:~^{}}'.format('done in {}s'.format(end-start), twidth))
    
