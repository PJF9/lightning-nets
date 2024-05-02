from typing import Tuple, Union, List, Generator, Iterator


# Define the aliases that tensor uses
dtype = Union[int, float]
NestedList = Union[dtype, List['NestedList']]


class Tensor:
    '''
    A class to represent tensors of any dimension in Python.
    '''
    def __init__(self,
        data: NestedList=None,
        dimensions: Tuple[int]=None,
        required_grad: bool=True,
        _children: Tuple['Tensor']=(),
        _op: str='',
        label: str=''
        ) -> None:
        '''
        Initializes a Tensor object with the specified dimensions.

        :param data: The content of the tensor
        :param dimensions: The dimensions of the tensor
        :param _children: The Tensor objects that produces this exact object
        :param required_grad: If the user want to compute gradients of this tensor
        :param _op: The operation which the object has been produced by the children (for visualization)
        :param label: The label of the object (for visualization)
        '''
        if (data is None) and (dimensions is None):
            self.dimensions = (0,)
            self.data = []
        elif (data is None):
            self.dimensions = dimensions
            self.data = self._initialize_data(dimensions)
        else:
            t_dimensions = self._find_dimensions(data)

            # if the user passes both data and dimensions, but the given dimensions and the data's dimensions don't match 
            if (dimensions is not None) and (dimensions != t_dimensions):
                raise ValueError('Tensor dimensions don\'t match')

            err = self._check_dimensions(data, t_dimensions)

            if err == -1:
                raise ValueError('Tensor dimensions don\'t match')
            
            self.dimensions = t_dimensions
            self.data = data

        self._op = _op
        self.label = label
        self._prev = set(_children) # for inceased speed
        self.required_grad = required_grad

        # The function that will compute the backpropagation        
        self._backward = lambda: None

        # Initialize the gradient of the List object
        self.grad = self._initialize_data(self.dimensions)

    def _initialize_data(self, dimensions: Tuple[int], value: dtype=0.0) -> Union[int, List[dtype]]:
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

    def _find_dimensions(self, data: Union[List[dtype], int]) -> Tuple[int]:
        '''
        Recursively finds the dimensions of the tensor

        :param data: The input tensor

        :return: A tuple representing the dimensions of the nested list
        '''
        if isinstance(data, int):
            return ()  # Base case: for a single integer, return an empty tuple
        else:
            return (len(data),) + self._find_dimensions(data[0])  # Recursively find dimensions of sub-lists

    def _check_dimensions(self, data: NestedList, dimensions: Tuple[int]) -> int:
        '''
        Check if all tensor dimensions match

        :param data: The tensor that its dimensions are going to be checked
        :param dimensions: The dimension that the tensor should have accross its values

        :return: The code that represents whether the tensor has the correct dimensions
        '''
        check_data = self._initialize_data(dimensions)

        def rec(t1, t2):
            if isinstance(t1, int) or isinstance(t2, int):
                return 0
        
            current1 = t1
            current2 = t2

            if len(current1) != len(current2):
                return -1
    
            for i in range(len(current1)):
                res = rec(current1[i], current2[i]) # check if every element of each dimension match
                if res == -1: # if find a dimension where the elements don't match, return
                    return -1

            return 0 # the dimensions match

        return rec(data, check_data)

    def _assign_value(self, value: dtype, *indices: Tuple[int]) -> None:
        '''
        Assigns a value to the specified indices in the tensor.

        :param value: The value to be assigned
        :param indices: Variable number of indices to access the element in the tensor
        '''
        current = self.data
        for index in indices[:-1]:
            current = current[index]
        current[indices[-1]] = value
    
    def _get_value(self, *indices: Tuple[int]) -> dtype:
        '''
        Retrieves the value from the specified indices in the tensor.

        :param indices: Variable number of indices to access the element in the tensor

        :return: The value at the specified indices
        '''
        current = self.data
        for index in indices:
            current = current[index]
        return current
    
    def _iterate_data(self, data: NestedList) -> Generator[dtype, None, None]:
        '''
        Recursively iterate over all elements of the tensor data

        :param data: The data to iterate over

        :return: A generator yielding elements of type dtype
        '''
        if isinstance(data, list):
            for sub_data in data:
                yield sub_data
        else:
            yield data
    
    def shape(self, index: int=None) -> Union[int, Tuple[int]]:
        '''
        Return the dimensions of the tensor

        :param index: Return a specific dimension

        :return: The dimensions of the tensor
        '''
        if index is None:
            return self.dimensions
        return self.dimensions[index]
    
    def tolist(self) -> List[dtype]:
        '''
        Convert the tensor to a Python list

        :return: The list version of the tensor
        '''
        return self.data
    
    def item(self) -> Union[dtype, None]:
        '''
        Convert a scalar tensor to intager or float

        :return: The resulting intager or float if tensor is a scalar, otherwise error
        '''
        if len(self.dimensions) == 0:
            return self.data
        raise ValueError(f'The given tensor is not a scalar, it has dimensions of {self.dimensions}')
    
    def __str__(self) -> str:
        '''
        Convert the tensor to string
        '''
        if self.required_grad:
            return f'Tensor({str(self.data)}, required_grad=True)'
        return f'Tensor({str(self.data)}, required_grad=False)'

    def __getitem__(self, indices: Union[int, Tuple[int]]) -> 'Tensor':
        '''
        Overloaded method to provide a more intuitive way of accessing elements in the tensor.

        :param indices: Indices to access the element in the tensor

        :return: The tesnor at the specified indices
        '''
        if isinstance(indices, tuple):
            return Tensor(
                data=self._get_value(*indices),
                required_grad=self.required_grad
            )
        else:
            return Tensor(
                data=self._get_value(indices),
                required_grad=self.required_grad
            )

    def __setitem__(self, indices: Union[int, Tuple[int]], value: dtype) -> None:
        '''
        Overloaded method to assign a value to the specified indices in the tensor.

        :param indices: Indices to access the element in the tensor
        :param value: The value to be assigned
        '''
        if isinstance(indices, tuple):
            self._assign_value(value, *indices)
        else:
            self._assign_value(value, indices)

    def __iter__(self) -> Iterator[dtype]:
        '''
        Iterate over all elements of the tensor

        :return: An iterator yielding elements of type dtype
        '''
        yield from self._iterate_data(self.data)
