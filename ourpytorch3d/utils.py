# import copy
# import inspect
# import warnings
from typing import Any, List, Optional, Tuple, TypeVar, Union

import numpy as np
import jittor as jt
import jittor.nn as nn

# from ..common.datatypes import Device, make_device

def parse_image_size(
    image_size: Union[List[int], Tuple[int, int], int]
) -> Tuple[int, int]:
    """
    Args:
        image_size: A single int (for square images) or a tuple/list of two ints.

    Returns:
        A tuple of two ints.

    Throws:
        ValueError if got more than two ints, any negative numbers or non-ints.
    """
    if not isinstance(image_size, (tuple, list)):
        return (image_size, image_size)
    if len(image_size) != 2:
        raise ValueError("Image size can only be a tuple/list of (H, W)")
    if not all(i > 0 for i in image_size):
        raise ValueError("Image sizes must be greater than 0; got %d, %d" % image_size)
    if not all(isinstance(i, int) for i in image_size):
        raise ValueError("Image sizes must be integers; got %f, %f" % image_size)
    return tuple(image_size)