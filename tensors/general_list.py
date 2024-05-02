from typing import Tuple, Union, List


class Tensor:
    '''
    A class to represent tensors of any dimension in Python.
    '''
    def __init__(self, dimensions: Tuple[int]) -> None:
        '''
        Initializes a Tensor object with the specified dimensions.

        :param dimensions: The dimensions of the tensor
        '''
        self.dimensions = dimensions
        self.data = self._initialize_data(dimensions)
    
    def _initialize_data(self, dimensions: Tuple[int], value: Union[int, float]=0) -> Union[int, List[Union[int, float]]]:
        '''
        Recursively initializes the data structure for the tensor.

        :param dimensions: The dimensions of the tensor
        :param value: The default value that the tensor will be filled with

        :return: The initialized data structure
        '''
        if len(dimensions) == 0:
            return value  # Scalars have no dimensions, so data is just a number
        else:
            return [self._initialize_data(dimensions[1:], value) for _ in range(dimensions[0])]

    def _assign_value(self, value: Union[int, float], *indices: Tuple[int]) -> None:
        '''
        Assigns a value to the specified indices in the tensor.

        :param value: The value to be assigned
        :param indices: Variable number of indices to access the element in the tensor
        '''
        current = self.data
        for index in indices[:-1]:
            current = current[index]
        current[indices[-1]] = value
    
    def _get_value(self, *indices: Tuple[int]) -> Union[int, float]:
        '''
        Retrieves the value from the specified indices in the tensor.

        :param indices: Variable number of indices to access the element in the tensor

        :return: The value at the specified indices
        '''
        current = self.data
        for index in indices:
            current = current[index]
        return current
    
    def __str__(self) -> str:
        '''
        Convert the tensor to string
        '''
        return f'Tensor({str(self.data)})'
    
    def __getitem__(self, indices: Union[int, Tuple[int]]) -> Union[int, float]:
        '''
        Overloaded method to provide a more intuitive way of accessing elements in the tensor.

        :param indices: Indices to access the element in the tensor

        :return: The value at the specified indices
        '''
        if isinstance(indices, tuple):
            return self._get_value(*indices)
        else:
            return self._get_value(indices)

    def __setitem__(self, indices: Union[int, Tuple[int]], value: Union[int, float]) -> None:
        '''
        Overloaded method to assign a value to the specified indices in the tensor.

        :param indices: Indices to access the element in the tensor
        :param value: The value to be assigned
        '''
        if isinstance(indices, tuple):
            self._assign_value(value, *indices)
        else:
            self._assign_value(value, indices)
