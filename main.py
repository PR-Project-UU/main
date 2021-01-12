from arguments import args
from logging import getLogger

def not_implemented():
    log = getLogger()
    log.info('This mode (%s) is yet to be implemented', args.mode[0])

def generate():
    '''Runs the generation cycle'''
    row = 0

mode_table = {
    'generate': not_implemented, #generate,
    'preprocess': not_implemented,
    'predict': not_implemented,
    'train': not_implemented,
}

mode_table[args.mode[0]]()