from argparse import ArgumentParser
from model_v2 import get_generator
import numpy as np
from os import listdir, path
from pickle import load
from random import shuffle
from tensorflow.keras import Model
from tensorflow.keras.layers import GlobalAveragePooling2D
from typing import Tuple
from fid import get_fid

ap = ArgumentParser()
sp = ap.add_subparsers(dest='parser', required=True)
cp = sp.add_parser('compare')
tp = sp.add_parser('test')
cp.add_argument('images', metavar='PATH', type=str, nargs=2, default=[None, None], help='The two images (as pickles) to calculate the FID for')
tp.add_argument('test_path', metavar='PATH', type=str, nargs=1, help='Run a random image generation test on pickles in the folder')
tp.add_argument('generator', metavar='PATH', type=str, nargs=1, help='The path to the model pickle for the generator')
tp.add_argument('test_count', metavar='INT', type=int, nargs=1, default=[None], help='Limit the amount of random tests to a specified integer')

args = ap.parse_args()

def exists_next(file_path: str, file: str) -> Tuple[bool, str]:
    '''Check if the next file exists

    Args:
        file_path (str): The path the file is located
        file (str): The file name to check from

    Returns:
        Tuple[bool, str]: A tuple with whether or not the next file exists, and the full path to the next file
    '''

    year_ext, tile, coordinates = [i[::-1] for i in file[::-1].split('-', 2)]
    year, ext = year_ext.split('.')
    next_path = path.join(file_path, '%s-%s-%s.%s' % (coordinates, tile, int(year) + 1, ext))

    return path.exists(next_path), next_path

def prep(image: np.ndarray):
    if len(image.shape) == 3:
        image = np.expand_dims(image)

    image = (image * 0.5 + 0.5) * 255

    return np.moveaxis(image, 3, 1)

if args.parser == 'compare':
    with open(args.images[0], 'rb') as f:
        image_1 = prep(load(f))

    with open(args.images[1], 'rb') as f:
        image_2 = prep(load(f))

    score = get_fid(image_1, image_2)

    print('Calulated FID: %.3f' % score)
else:
    files = [(ff[0], ff[1][1]) for ff in [(path.join(args.test_path[0], f), exists_next(args.test_path[0], f)) for f in listdir(args.test_path[0]) if f.split('.')[-1] == 'pickle'] if ff[1][0]]
    count = min(len(files), args.test_count[0] or len(files))

    with open(args.generator[0], 'rb') as f:
        generator_weights, _ = load(f)

    generator = get_generator()
    
    generator.set_weights(generator_weights)
    shuffle(files)

    inputs = np.empty([0, 64, 64, 3])
    targets = np.empty([0, 64, 64, 3])

    for inp_file, tar_file in files[:count]:
        with open(inp_file, 'rb') as f:
            inp = load(f)

        with open(tar_file, 'rb') as f:
            tar = load(f)

        if (np.isnan(inp).sum() / inp.size < 0.1).all():
            inp = np.nan_to_num(inp)
        else:
            count -= 1
            continue

        if (np.isnan(tar).sum() / tar.size < 0.1).all():
            tar = np.nan_to_num(tar)
        else:
            count -= 1
            continue

        inputs = np.concatenate([inputs, inp[None]], axis=0)
        targets = np.concatenate([targets, tar[None]], axis=0)

    results = generator.predict_on_batch(inputs)
    fid = get_fid(prep(results), prep(targets))

    print('Fid score: %.3f' % fid)
