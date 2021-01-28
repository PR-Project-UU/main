from arguments import args
from logging import getLogger

def not_implemented():
    log = getLogger()
    log.info('This mode (%s) is yet to be implemented', args.mode[0])

def create():
    '''Runs the prediction cycle'''
    from create import Creator
    from sys import stdin
    from os import path
    from matplotlib import pyplot as plt
    import numpy as np

    creator = Creator(args.model[0], args.save_path[0])
    log = getLogger('creator')

    if args.load_path[0] is None or not args.load_path[0].split('.')[-1] in ['pickle', 'tif']:
        try:
            lines = int(stdin.readline())
        except ValueError:
            log.critical('The first value of stdin should be the number of input files when no "--load_path" is provided.')
            exit(1)

        paths = []
        
        for _ in range(lines):
            paths.append(stdin.readline()[:-1])
        
        for path in paths:
            image = creator.create(path, args.meta)
            filename = '.'.join(path.split('/')[-1].split('.')[:-1]) + '.png'

            plt.imsave(path.join(args.save_path[0], filename), image, vmin=0, vmax=1)
    else:
        image = creator.create(args.load_path[0], args.meta)
        filename = '.'.join(args.load_path[0].split('/')[-1].split('.')[:-1]) + '.png'

        plt.imsave(path.join(args.save_path[0], filename), image, vmin=0, vmax=1)

    log.info('We did it')

def generate():
    '''Runs the generation cycle'''
    # Import here to prevent slowdown in different modes
    from generate import Generator

    gen = Generator()
    gen.run(args.save_path[0], args.delete)

def preprocess():
    '''Runs the preprocessing cycle'''
    # Import here to prevent slowdown in different modes
    from image import preprocess
    from os import path, listdir

    files = [path.join(args.load_path[0], f) for f in listdir(args.load_path[0]) if path.isfile(path.join(args.load_path[0], f))]

    for file in files:
        if file.split('.')[-1] == 'tif':
            preprocess(file, args.save_path[0], args.delete, not args.no_overwrite)

    getLogger('preprocess').info('Preprocessed %s files to "%s"', len(files), args.save_path[0] or args.load_path[0])

# def train():
#     '''Runs the training cycle'''
#     # Import here to prevent slowdown in different modes
#     from train import Dataset, Trainer

#     ds = Dataset(args.load_path[0])
#     trainer = Trainer(ds, args.model[0], args.save_path[0], args.epochs[0])
#     trainer.fit()

def train():
    '''Runs the training cycle'''
    # Import here to prevent slowdown in different modes
    from model_v2 import Trainer
    batch_size = 5
    batches_per_epoch = 100
    trainer = Trainer(args.load_path[0], args.model[0], args.save_path[0], args.epochs[0], batch_size, batches_per_epoch)
    trainer.fit()

mode_table = {
    'generate': generate,
    'preprocess': preprocess,
    'predict': create,
    'train': train,
}

mode_table[args.mode[0]]()