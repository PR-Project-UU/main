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

    padding = floor((min(len(image[0]), len(image[0][0])) - 64) / 2)
    cropped = np.zeros((image.shape[0], 64, 64))
    
    for i, layer in enumerate(image):
        cropped[i] = layer[padding:(padding + 64), padding:(padding + 64)]

    return cropped

def normalize(image: np.ndarray) -> np.ndarray:
    image /= np.nanmax(image)
    image = (image * 3000.0) / 1500.0 - 1.0
    return image.astype('float32')

def denormalize(output: np.ndarray) -> np.ndarray:
    image = output * 0.5 + 0.5
    print('%.5f -> %.5f' % (np.nanmax(output), np.nanmax(image)))
    return image

def preprocess(file: str, save_path: str = None, delete_original: bool = False, overwrite: bool = True, crop_image: bool = True, return_value: bool = False):
    '''Preprocesses an image based on its filename

    Args:
        file (str): The filename (and path) to load the image from.
        save_path (str, optional): The path to save the new image to, if None saves it to the same directory. Defaults to None.
        delete_original (bool, optional): Deletes the original image after preprocessing it. Defaults to False.
        overwrite (bool, optional): Overwrite the file if it already exists. Defaults to True.
        crop_image (bool, optional): Crops the image to 64x64 size from center. Defaults to True.
        return_value (bool, optional): Returns the np array instead of pickling it to file. Defaults to False.

    Returns:
        (np.ndarray | None): Returns the image if return_value is True.
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
    if not path.exists(save_path) and not return_value:
        log.error('Save path "%s" is invalid or does not exist', save_path)
        return

    save_file = save_path + '.'.join(file.split('/')[-1].split('.')[:-1]) + '.pickle'

    # Check if the file does already exist
    if not overwrite and path.exists(save_file) and not return_value:
        log.error('Save file "%s" already exists and overwrite is disabled', save_file)
        return

    # Open the file
    with rasterio.open(file) as f:
        image = f.read()

    # Process the image (crop and normalize [-1.0, 1.0])
    if crop_image:
        image = np.moveaxis(normalize(crop(image)), 0, -1)
    else:
        image = np.moveaxis(normalize(image), 0, -1)

    # Save the image
    if not return_value:
        with open(save_file, 'wb') as f:
            pickle.dump(image, f)

    # Delete the original if so desired
    if delete_original:
        remove(file)

    if not return_value:
        log.info('Preprocessed image "%s" to "%s"', file, save_path)
    else:
        log.info('Preprocessed image "%s"', file)
        return image