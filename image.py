from logging import getLogger
from math import floor
import numpy as np
from os import path, remove
import pickle
import rasterio

def crop(image: np.ndarray):
    '''Crops a layerd image into 66x66 square layered image (in place)

    Args:
        image (np.ndarray): The image to crop

    Returns:
        (np.ndarray): The cropped image
    '''

    padding = floor((min(len(image[0]), len(image[0][0])) - 66) / 2)
    cropped = np.zeros((image.shape[0], 66, 66))
    
    for i, layer in enumerate(image):
        cropped[i] = layer[padding:(padding + 66), padding:(padding+66)]

    return cropped

def preprocess(file: str, save_path: str = None, delete_original: bool = False, overwrite: bool = True):
    '''Preprocesses an image based on its filename

    Args:
        file (str): The filename (and path) to load the image from.
        save_path (str, optional): The path to save the new image to, if None saves it to the same directory. Defaults to None.
        delete_original (bool, optional): Deletes the original image after preprocessing it. Defaults to False.
        overwrite (bool, optional): Overwrite the file if it already exists
    '''
    log = getLogger('preprocessing')

    # Establish the save path
    if save_path is None:
        file_path = file.split('/')

        if len(file_path) == 1:
            save_path = './'
        else:
            save_path = '/'.join(file_path[:-1]) + '/'

    # Check if the save_path is valid
    if not path.exists(save_path):
        log.error('Save path "%s" is invalid or does not exist', save_path)
        return

    save_file = save_path + '.'.join(file.split('/')[-1:].split('.')[:-1]) + '.pickle'

    # Check if the file does already exist
    if not overwrite and path.exists(save_file):
        log.error('Save file "%s" already exists and overwrite is disabled', save_file)
        return

    # Open the file
    with rasterio.open(file) as f:
        image = f.read()

    # Process the image
    crop(image)

    # Save the image
    with open(save_file, 'wb') as f:
        pickle.dump(image, f)

    # Delete the original if so desired
    if delete_original:
        remove(file)

    log.info('Preprocessed image "%s" to "%s"', file, save_path)