from arguments import args
from logging import getLogger

def not_implemented():
    log = getLogger()
    log.info('This mode (%s) is yet to be implemented', args.mode[0])

def generate():
    '''Runs the generation cycle'''
    # Import here to prevent slowdown in different modes
    from generate import Generator

    gen = Generator()
    gen.run(args.save_path[0], args.delete)

mode_table = {
    'generate': generate,
    'preprocess': not_implemented,
    'predict': not_implemented,
    'train': not_implemented,
}

mode_table[args.mode[0]]()