from ee import Image, Reducer # NOTE: This won't work if ee isn't already initialized, because Reducer won't exist
from math import ceil, sqrt
from typing import Tuple

def cloudmask_L457(image: Image) -> Image:
    '''Function to mask clouds based on the pixel_qa band of the LandsatSR data

    Args:
        image (Image): The input image to mask on

    Returns:
        Image: Cloudmasked output image
    '''        
    qa = image.select('pixel_qa')
    
    # If the cloud bit (5) is set and the cloud confidence (7) is high
    # or the cloud shadow bit is set (3), then it's a bad pixel.
    cloud = qa.bitwiseAnd(1 << 5).And(qa.bitwiseAnd(1 << 7)).Or(qa.bitwiseAnd(1 << 3))
    
    # Remove edge pixels that don't occur in all bands
    mask = image.mask().reduce(Reducer.min())
    
    return image.updateMask(cloud.Not()).updateMask(mask)

def spiralIndex(index) -> Tuple[int, int]:
    '''Creates a coordinate on a 2D grid from an index (spiraled outward from the center at [0,0])

    Args:
        index (int): The index to get the location for

    Returns:
        Tuple[int, int]: The coordinates requested
    '''

    index += 1
    k = ceil((sqrt(index) - 1) / 2)
    t = 2 * k + 1
    m = t ** 2
    t -= 1

    if index >= m - t:
        return (k - (m - index), k)
    else:
        m = m - t

    if index >= m - t:
        return (-k, k - (m - index))
    else:
        m = m - t

    if index >= m - t:
        return (-k + (m - index), -k)
    
    return (k, -k + (m - index - t))
