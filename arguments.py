from argparse import ArgumentParser
from logging import basicConfig, getLogger, StreamHandler, Formatter
from sys import argv, stdout

parser = ArgumentParser()
parser.add_argument('-e', '--delete', action='store_true', default=False, help='Delete files after downloading or preprocessing')
parser.add_argument('-c', '--epochs', nargs=1, default=[150], type=int, help='The total amount of epochs to train for')
parser.add_argument('-p', '--load-path', nargs=1, default=['./data/raw/'], type=str, help='The path to load images from')
parser.add_argument('-l', '--log', nargs='?', default='info', const='debug', type=str, choices=['debug', 'info', 'warning', 'error', 'critical'], help='Sets the console\'s output level')
parser.add_argument('-m', '--mode', nargs=1, default=['generate'], type=str, choices=['generate', 'preprocess', 'train', 'predict'], help='The mode to run the program in')
parser.add_argument('-n', '--no-overwrite', action='store_true', default=False, help='Prevent overwriting files that already exist')
parser.add_argument('-o', '--model', nargs=1, default=[None], type=str, help='The name of the model to load, use, or save to')
#parser.add_argument('--no-download', action='store_true', default=False, help='Don\'t download after generating')
parser.add_argument('-s', '--save-path', nargs=1, default=['./data/raw'], type=str, help='The path to save downloaded files to')
#parser.add_argument('-t', '--timeout', nargs=1, default=[None], type=int, help='The time to wait for downloads to be ready in seconds')

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
console_handler.setFormatter(Formatter('%(asctime)s %(levelname)8s %(name)20s > %(message)s', datefmt='%H:%M:%S'))
main_logger.addHandler(console_handler)

log = getLogger('arguments')
log.debug('Starting program with arguments "%s"', ' '.join(argv))
