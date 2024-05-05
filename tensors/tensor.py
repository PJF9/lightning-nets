import math
from functools import reduce
from operator import mul
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
        if isinstance(data, (int, float)):
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
            if isinstance(t1, (int, float)) or isinstance(t2, (int, float)):
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
        for i in range(self.dimensions[0]):
            yield self[i]

    @staticmethod
    def _flatten(data: NestedList) -> List[dtype]:
        '''
        Convert the multidimensional tensor's elements into an 1d list

        :param data: The elements of the multidimensional tensor

        :return: The 1d list
        '''
        def _flatten_rec(data: NestedList) -> Generator[dtype, None, None]:
            for item in data:
                if isinstance(item, list):
                    yield from _flatten_rec(item)
                else:
                    yield item

        flattened_data = list(_flatten_rec(data))  # Convert generator to list

        return flattened_data

    @staticmethod
    def _reshape_flatten(data: NestedList, new_dims: Tuple[int]) -> NestedList:
        '''
        Change flatten's tensors dimensions

        :param data: The elements of the flattened tensor
        :param new_dims: The dimensions the flatten tensor is going to have

        :return: The multidimensional tensor's elements
        '''
        j = 0
        def _reshape(data: NestedList, new_dimensions: Tuple[int]) -> NestedList:
            nonlocal j

            if len(new_dimensions) == 0:
                ret = data[j]
                j += 1
                return ret
            else:
                return [_reshape(data, new_dimensions[1:]) for _ in range(new_dimensions[0])]

        return _reshape(data, new_dims)
    
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

    def zero_grad(self) -> None:
        '''
        Reset the gradient of the tensor
        '''
        self.grad = self._initialize_data(self.dimensions)

    def __str__(self) -> str:
        '''
        Convert the tensor to string
        '''
        if self.required_grad:
            return f'Tensor({str(self.data)}, required_grad=True)'
        return f'Tensor({str(self.data)}, required_grad=False)'
    
    def __repr__(self) -> str:
        '''
        Convert the tensor to string
        '''
        return str(self)
    
    def __len__(self) -> int:
        '''
        Return the length of the first dimension of the tensor

        :return: The length of the tensor
        '''
        if isinstance(self.data, list):
            return len(self.data)
        return 0

    def __getitem__(self, indices: Union[int, Tuple[int]]) -> 'Tensor':
        '''
        Overloaded method to provide a more intuitive way of accessing elements in the tensor.

        :param indices: Indices to access the element in the tensor

        :return: The tesnor at the specified indices
        '''
        if isinstance(indices, tuple):
            res_obj = Tensor(
                data=self._get_value(*indices),
                required_grad=self.required_grad
            )
            res_obj.grad = self.grad(*indices)
        else:
            res_obj = Tensor(
                data=self._get_value(indices),
                required_grad=self.required_grad
            )
            res_obj.grad = self.grad[indices]

        return res_obj

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

    def __add__(self, sec_obj: 'Tensor') -> 'Tensor':
        '''
        Overloaded method to perform element-wise addition of two tensors.

        :param sec_obj: The second tensor to add

        :return: The result tensor after addition
        '''
        assert self.dimensions[-1] == sec_obj.dimensions[-1], f'Tensor dimensions don\'t match for addition: {self.shape()}, and {sec_obj.shape()}'

        add_on_last = False
        def _produce_res() -> Tensor:
            '''
            The function that will return the result of the addition
            '''
            nonlocal add_on_last
            # Flattening the tensors' elements
            flatten_self_data = Tensor._flatten(self.data)
            flatten_sec_data = Tensor._flatten(sec_obj.data)

            if (reduce(mul, sec_obj.dimensions, 1) == sec_obj.dimensions[-1]) and (len(flatten_self_data) != len(flatten_sec_data)):
                # extending the small tensor: for adding on the last dimension
                add_on_last = True
                flatten_sec_data = flatten_sec_data * self.dimensions[-1]

            if len(flatten_self_data) != len(flatten_sec_data):
                raise ValueError(f'Tensors cannot be added together with dimensions {self.dimensions}, and {sec_obj.dimensions}')

            if self.required_grad:
                return Tensor(
                    data=Tensor._reshape_flatten([el_1 + el_2 for el_1, el_2 in zip(flatten_self_data, flatten_sec_data)], self.dimensions),
                    required_grad=True,
                    _children=(self, sec_obj),
                    _op='+'
                )
            else:
                return Tensor(
                    data=Tensor._reshape_flatten([el_1 + el_2 for el_1, el_2 in zip(flatten_self_data, flatten_sec_data)], self.dimensions),
                    required_grad=False
                )
        
        res_obj = _produce_res()

        def _backward() -> None:
            '''
            Updating the gradients of the children Tensors for addition operation
            '''
            nonlocal add_on_last

            flatten_self_grad = Tensor._flatten(self.grad)
            flatten_sec_grad = Tensor._flatten(sec_obj.grad)
            flatten_res_grad = Tensor._flatten(res_obj.grad)

            if add_on_last:
                flatten_sec_grad = flatten_sec_grad * self.dimensions[-1]

            for i in range(len(flatten_self_grad)):
                flatten_self_grad[i] += flatten_res_grad[i]
                if i >= self.dimensions[-1] and add_on_last:
                    continue
                flatten_sec_grad[i] += flatten_res_grad[i]

            self.grad = Tensor._reshape_flatten(flatten_self_grad, self.dimensions)
            if add_on_last:
                sec_obj.grad = Tensor._reshape_flatten(flatten_sec_grad[:self.dimensions[-1]], sec_obj.dimensions)
            else:
                sec_obj.grad = Tensor._reshape_flatten(flatten_sec_grad, sec_obj.dimensions)

        res_obj._backward = _backward

        return res_obj

    def __neg__(self) -> 'Tensor':
        '''
        Implement the -self operation with the tensor

        :return: The result of the negation
        '''
        def negeate_tensor(t):
            if isinstance(t, list):
                return [negeate_tensor(sub_t) for sub_t in t]
            else:
                return -t
        
        if self.required_grad:
            res_obj = Tensor(
                data=negeate_tensor(self.data),
                required_grad=True,
                _children=(self,),
                _op='-'
            )
        else:
            res_obj = Tensor(
                data=negeate_tensor(self.data),
                required_grad=False,
            )

        def _backward() -> None:
            '''
            Updating the gradients of the children Tensors for addition operation
            '''
            def update_gradients(t, res_t):
                if isinstance(t.data, int):
                    return

                if isinstance(t.data, list):
                    for sub_t, sub_res_t in zip(t, res_t):
                        update_gradients(sub_t, sub_res_t)

                if not isinstance(t.data[0], list):
                    for i in range(len(t.data)):
                        t.grad[i] += -1 * res_t.grad[i]

            update_gradients(self, res_obj)

        res_obj._backward = _backward

        return res_obj

    def __sub__(self, sec_obj: 'Tensor') -> 'Tensor':
        '''
        Implement the __sub__ method for tensor

        :param sec_obj: The second operand of the subtraction

        :return: The result opf the subtraction
        '''
        return self + (-sec_obj)
    
    def __mul__(self, sec_obj: 'Tensor') -> 'Tensor':
        '''
        Implement the multiplication operator between two tensors

        :param sec_obj: The second operand of the multiplication

        :return: The result of the multiplication
        '''
        assert self.dimensions[-1] == sec_obj.dimensions[-1], f'Tensor dimensions don\'t match for multiplication: {self.shape()}, and {sec_obj.shape()}'

        mul_on_last = False
        flatten_self_data: Tensor
        flatten_sec_data: Tensor
        def _produce_res() -> Tensor:
            '''
            The function that will return the result of the addition
            '''
            nonlocal mul_on_last
            nonlocal flatten_self_data
            nonlocal flatten_sec_data
            # Flattening the tensors' elements
            flatten_self_data = Tensor._flatten(self.data)
            flatten_sec_data = Tensor._flatten(sec_obj.data)

            # the first if: shape of (1, 1, ..., 1, n) | second: the two arrays don't have the same number of elements
            if (reduce(mul, sec_obj.dimensions, 1) == sec_obj.dimensions[-1]) and (len(flatten_self_data) != len(flatten_sec_data)):
                # extending the small tensor: for adding on the last dimension
                mul_on_last = True
                flatten_sec_data = flatten_sec_data * self.dimensions[-1]

            if len(flatten_self_data) != len(flatten_sec_data):
                raise ValueError(f'Tensors cannot be added together with dimensions {self.dimensions}, and {sec_obj.dimensions}')

            if self.required_grad:
                return Tensor(
                    data=Tensor._reshape_flatten([el_1 * el_2 for el_1, el_2 in zip(flatten_self_data, flatten_sec_data)], self.dimensions),
                    required_grad=True,
                    _children=(self, sec_obj),
                    _op='+'
                )
            else:
                return Tensor(
                    data=Tensor._reshape_flatten([el_1 * el_2 for el_1, el_2 in zip(flatten_self_data, flatten_sec_data)], self.dimensions),
                    required_grad=False
                )
        
        res_obj = _produce_res()

        def _backward() -> None:
            '''
            Updating the gradients of the children Tensors for addition operation
            '''
            nonlocal mul_on_last

            flatten_self_grad = Tensor._flatten(self.grad)
            flatten_sec_grad = Tensor._flatten(sec_obj.grad)
            flatten_res_grad = Tensor._flatten(res_obj.grad)

            if mul_on_last:
                flatten_sec_grad = flatten_sec_grad * self.dimensions[-1]

            for i in range(len(flatten_self_grad)):
                flatten_self_grad[i] += flatten_sec_data[i] * flatten_res_grad[i]
                if i >= self.dimensions[-1] and mul_on_last:
                    continue
                flatten_sec_grad[i] += flatten_sec_data[i] * flatten_res_grad[i]

            self.grad = Tensor._reshape_flatten(flatten_self_grad, self.dimensions)
            if mul_on_last:
                sec_obj.grad = Tensor._reshape_flatten(flatten_sec_grad[:self.dimensions[-1]], sec_obj.dimensions)
            else:
                sec_obj.grad = Tensor._reshape_flatten(flatten_sec_grad, sec_obj.dimensions)

        res_obj._backward = _backward

        return res_obj

    def __pow__(self, exp: Union[int, float]) -> 'Tensor':
        '''
        Implement the self**exp operator for the Tensor

        :parap exp: The exponent of the power operation

        :return: The result of the power operation
        '''
        def pow_tensor(t):
            if isinstance(t, list):
                return [pow_tensor(sub_t) for sub_t in t]
            else:
                return t**exp
        
        if self.required_grad:
            res_obj = Tensor(
                data=pow_tensor(self.data),
                _children=(self,),
                _op='**'
            )
        else:
            res_obj = Tensor(
                data=pow_tensor(self.data)
            )

        def _backward() -> None:
            '''
            Updating the gradients of the children Tensors for addition operation
            '''
            def update_gradients(t, res_t):
                if isinstance(t.data, (int, float)):
                    return

                if isinstance(t.data, list):
                    for sub_t, sub_res_t in zip(t, res_t):
                        update_gradients(sub_t, sub_res_t)

                if not isinstance(t.data[0], list):
                    for i in range(len(t.data)):
                        t.grad[i] += exp*(t.data[i]**(exp-1)) * res_t.grad[i]

            update_gradients(self, res_obj)
            
        res_obj._backward = _backward

        return res_obj
    
    def __truediv__(self, sec_obj: 'Tensor') -> 'Tensor':
        '''
        Implement the division operation between two tensors

        :param sec_obj: The second operand of the division

        :return: The result of the devision
        '''
        return self * (sec_obj ** -1)

    def exp(self) -> 'Tensor':
        '''
        Implement the e^X operation for the tensors

        :return: The result of the e^X function
        '''
        def exp_tensor(t):
            if isinstance(t, list):
                return [exp_tensor(sub_t) for sub_t in t]
            else:
                return math.exp(t)
        
        if self.required_grad:
            res_obj = Tensor(
                data=exp_tensor(self.data),
                _children=(self,),
                _op='**'
            )
        else:
            res_obj = Tensor(
                data=exp_tensor(self.data)
            )

        def _backward() -> None:
            '''
            Updating the gradients of the children Tensors for addition operation
            '''
            def update_gradients(t, res_t):
                if isinstance(t.data, (int, float)):
                    return

                if isinstance(t.data, list):
                    for sub_t, sub_res_t in zip(t, res_t):
                        update_gradients(sub_t, sub_res_t)

                if not isinstance(t.data[0], list):
                    for i in range(len(t.data)):
                        t.grad[i] += t.data[i] * res_t.grad[i]

            update_gradients(self, res_obj)
            
        res_obj._backward = _backward

        return res_obj
    
    def tanh(self) -> 'Tensor':
        '''
        Implement the tanh function for Lists

        :return: The result of the tanh function
        '''
        def tanh_tensor(t):
            if isinstance(t, list):
                return [tanh_tensor(sub_t) for sub_t in t]
            else:
                return (math.exp(2 * t) - 1) / (math.exp(2 * t) + 1)
        
        if self.required_grad:
            res_obj = Tensor(
                data=tanh_tensor(self.data),
                _children=(self,),
                _op='**'
            )
        else:
            res_obj = Tensor(
                data=tanh_tensor(self.data)
            )

        def _backward() -> None:
            '''
            Updating the gradients of the children Tensors for addition operation
            '''
            def update_gradients(t, res_t):
                if isinstance(t.data, (int, float)):
                    return

                if isinstance(t.data, list):
                    for sub_t, sub_res_t in zip(t, res_t):
                        update_gradients(sub_t, sub_res_t)

                if not isinstance(t.data[0], list):
                    for i in range(len(t.data)):
                        t.grad[i] += (1- res_t.data[i]**2) * res_t.grad[i]

            update_gradients(self, res_obj)
            
        res_obj._backward = _backward

        return res_obj

    def relu(self) -> 'Tensor':
        '''
        Implement the ReLU function for Lists

        :return: The result of the ReLU function
        '''
        def relu_tensor(t):
            if isinstance(t, list):
                return [relu_tensor(sub_t) for sub_t in t]
            else:
                return t if t > 0 else 0
        
        if self.required_grad:
            res_obj = Tensor(
                data=relu_tensor(self.data),
                _children=(self,),
                _op='**'
            )
        else:
            res_obj = Tensor(
                data=relu_tensor(self.data)
            )

        def _backward() -> None:
            '''
            Updating the gradients of the children Tensors for addition operation
            '''
            def update_gradients(t, res_t):
                if isinstance(t.data, (int, float)):
                    return

                if isinstance(t.data, list):
                    for sub_t, sub_res_t in zip(t, res_t):
                        update_gradients(sub_t, sub_res_t)

                if not isinstance(t.data[0], list):
                    for i in range(len(t.data)):
                        t.grad[i] += (t.data[i] > 0) * res_t.grad[i]

            update_gradients(self, res_obj)
            
        res_obj._backward = _backward

        return res_obj

    def backward(self) -> None:
        '''
        Implement backpropagation for the tensor class
        '''
        topo = []
        visited = set()
        def _build_topo(v) -> None:
            '''
            Apply topological sorting to the children of a tensor

            :param v: The object that we want to apply topological sorting
            the its children
            '''
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    _build_topo(child)
                topo.append(v)

        _build_topo(self)

        # Set the initial grad
        self.grad = self._initialize_data(self.dimensions, 1.0)

        # Apply topological sorted to compute the _backward method of every children
        for n in reversed(topo):
            n._backward()
