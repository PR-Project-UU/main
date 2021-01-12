from argparse import ArgumentParser
from logging import basicConfig, getLogger, StreamHandler
from sys import argv, stdout

parser = ArgumentParser()
parser.add_argument('-e', '--delete', action='store_true', default=False, help='Delete files after downloading or preprocessing')
parser.add_argument('-l', '--log', nargs='?', default='info', const='debug', type=str, choices=['debug', 'info', 'warning', 'error', 'critical'], help='Sets the console\'s output level')
parser.add_argument('-m', '--mode', nargs=1, default='generate', type=str, choices=['generate', 'preprocess', 'train', 'predict'], help='The mode to run the program in')
parser.add_argument('-s', '--savepath', nargs=1, default='./', type=str, help='The path to save downloaded files to')

args = parser.parse_args()

# Pre-process some args
log_table = {
    'debug': 10,
    'info': 20,
    'warning': 30,
    'error': 40,
    'critical': 50
}
args.log = log_table[args.log]

# Set the default config for logging
basicConfig(filename='debug.log',
            filemode='a',
            format='%(asctime)s %(levelname)8s %(name)20s > %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            level=0)

main_logger = getLogger()
console_handler = StreamHandler()
console_handler.setLevel(args.log)
console_handler.setStream(stdout)
main_logger.addHandler(console_handler)

log = getLogger('arguments')
log.debug('Starting program with arguments "%s"', ' '.join(argv))
