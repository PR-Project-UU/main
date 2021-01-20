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

def train():
    '''Runs the training cycle'''
    # Import here to prevent slowdown in different modes
    from train import Dataset, Trainer

    ds = Dataset(args.load_path[0])
    trainer = Trainer(ds, args.model[0], args.save_path[0], args.epochs[0])
    trainer.fit()

mode_table = {
    'generate': generate,
    'preprocess': preprocess,
    'predict': not_implemented,
    'train': train,
}

mode_table[args.mode[0]]()