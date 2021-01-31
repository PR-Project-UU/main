from image import preprocess, denormalize
from logging import getLogger
from math import ceil
from model import get_generator
from util import normalize_meta
import numpy as np
from os import path
import pickle
import tensorflow as tf

class Creator():
    generator = get_generator()
    log = getLogger('creator')
    save_path: str = None

    def __init__(self, model: str, save_path: str):
        if not path.exists(model):
            self.log.critical('Cannot find model "%s"', model)
            exit(1)

        with open(model, 'rb') as f:
            weights, _ = pickle.load(f)

        self.generator.set_weights(weights)       
        self.save_path = save_path 

    def create(self, source: str, to_pickle: bool = False):
        source_ext = source.split('.')[-1]

        if not path.exists(source) or not source_ext in ['pickle', 'tif']:
            self.log.error('Cannot find TIF image or pickle at "%s"', source)
            return

        if source_ext == 'tif':
            image = preprocess(source, crop_image = False, return_value = True)
            image = np.nan_to_num(image)
        else:
            with open(source, 'rb') as f:
                image = pickle.load(f)

        if image.shape[0] < 64 or image.shape[1] < 64:
            self.log.error('Image shape needs to be at least 64 by 64 pixels.')
            return
        elif image.shape[0] > 64 or image.shape[1] > 64:
            self.log.info("Original image size %d by %d", image.shape[1], image.shape[0])
            tiles = self.tile(image)
            self.log.info("Cut into %d tiles", len(tiles))
        else:
            tiles = [image[None]]

        results = []

        for tile in tiles:
            if to_pickle:
                results.append(np.squeeze(self.generator([tile]).numpy()))
            else:
                results.append(denormalize(np.squeeze(self.generator([tile]).numpy())))

        if len(results) > 1:
            stitched = self.stitch(results, image.shape)
            self.log.info('Stiched image dimensions %d by %d', stitched.shape[1], stitched.shape[0])
            return stitched
        else:
            return results[0]

    @staticmethod
    def tile(image: np.ndarray):
        columns = ceil(image.shape[1] / 64)
        rows = ceil(image.shape[0] / 64)
        tiles = [None] * (rows * columns)

        for i in range(len(tiles)):
            x = (i % columns) * 64 if (i % columns) + 1 < columns else image.shape[1] - 64
            y = (i // rows) * 64 if (i // rows) + 1 < rows else image.shape[0] - 64

            tiles[i] = image[None, y : y + 64, x : x + 64]

        return tiles

    @staticmethod
    def stitch(tiles, target_shape):
        columns = ceil(target_shape[1] / 64)
        rows = ceil(target_shape[0] / 64)

        result = np.empty((0, target_shape[1], 3))
        overlap = [rows * 64 - target_shape[0], columns * 64 - target_shape[1]]

        for r in range(rows - 1):
            row = tiles[r * columns]

            for c in range(1, columns - 1):
                row = np.concatenate([row, tiles[r * columns + c]], axis = 1)

            row = np.concatenate([row, tiles[r * columns + columns - 1][:, overlap[1]:]], axis = 1)

            result = np.concatenate([result, row], axis = 0)

        row = tiles[(rows - 1) * columns][overlap[0]:,:]

        for c in range(1, columns - 1):
            row = np.concatenate([row, tiles[(rows - 1) * columns + c][overlap[0]:,:]], axis = 1)
        
        row = np.concatenate([row, tiles[(rows - 1) * columns + columns - 1][overlap[0]:, overlap[1]:]], axis = 1)
        
        return np.concatenate([result, row], axis = 0)
