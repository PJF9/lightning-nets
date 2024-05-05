from tensors.tensor import Tensor

import random
from typing import Tuple, Union, List


# Define the aliases that tensor uses
dtype = Union[int, float]


def randn(dimensions: Tuple[int], mode: str='gauss', seed: int=None) -> Tensor:
    '''
    The function that constructs a tensor and filling it with random elements

    :param dimensions: The dimensions of the returing tensor
    :param mode: The distribution of which the random values will be taken
    :param seed: The seed of the random number generator

    :return: The random initialized tensor
    '''
    if seed is not None:
        random.seed(seed)

    def _initialize_data(dimensions: Tuple[int]) -> Union[int, List[dtype]]:
        '''
        Recursively initializes a list with random data

        :param dimensions: The dimensions of the tensor

        :return: The initialized random list
        '''
        if len(dimensions) == 0:
            return random.gauss(0, 1) if mode == 'gauss' else random.uniform(-1, 1)
        else:
            return [_initialize_data(dimensions[1:]) for _ in range(dimensions[0])]

    random_list = _initialize_data(dimensions)

    return Tensor(random_list)
