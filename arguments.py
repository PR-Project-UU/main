from argparse import ArgumentParser
from logging import basicConfig

parser = ArgumentParser()
parser.add_argument('-d', '--debug', nargs='?', default='info', const='debug', type=str, choices=['debug', 'info', 'warning', 'error', 'critical'], help='Sets the console\'s output level')
parser.add_argument('-e', '--delete', action='store_true', default=False, help='Delete files from Google Drive after downloading')
parser.add_argument('-m', '--mode', nargs=1, default='generate', type=str, choices=['generate', 'train', 'predict'], help='The mode to run the program in')
parser.add_argument('-s', '--savepath', nargs=1, default='./', type=str, help='The path to save downloaded files to')

args = parser.parse_args()

# Pre-process some args
debug_table = {
    'debug': 10,
    'info': 20,
    'warning': 30,
    'error': 40,
    'critical': 50
}
args.debug = debug_table[args.debug]

# Set the default config for logging
basicConfig(filename='debug.log',
            filemode='a',
            format='%(asctime)s, %(msec)02d %(levelname)8s %(name)20s> %(message)s',
            datefmt='%H:%M:%S',
            level=args.debug)
