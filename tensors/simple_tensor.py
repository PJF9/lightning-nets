import math
from typing import List, Tuple, Union


class SimpleTensor:
    '''
    My replication of the Tensor class of pytorch to help me better understand how
    deep neural networks work.
    It only works for 1d lists
    '''
    def __init__(self, 
        data: List[float],
        _children: Tuple['SimpleTensor']=(),
        _op: str='',
        label: str='') -> None:
        '''
        Initialize the List class which would be a normal Python list with
        backpropagetion

        :param data: The content of the list
        :param _children: The List objects that produces the object
        :param _op: The operation which the object has been produced by the children
        :param label: The label of the object, for visualization
        '''
        self.data = data
        self._op = _op
        self.label = label
        self._prev = set(_children)

        # Initialize the gradient of the List object
        self.grad = [0.0 for _ in range(len(data))]

        # The function that will compute the backpropagation        
        self._backward = lambda: None

    def __repr__(self) -> str:
        '''
        Implementation of the __repr__ method to provide the debug capability

        :return: The debug string representation of the string
        '''
        if self.label != '':
            return f'List{self.data}, label={self.label}'
        return f'List{self.data}'

    def __str__(self) -> str:
        '''
        Implementation of the __str__ method to convert the object to string

        :return: The string representation of the List
        '''
        return str(self.data)
    
    def __add__(self, sec_obj: 'SimpleTensor') -> 'SimpleTensor':
        '''
        Implement the __add__ method to add two List

        :param sec_obj: The object that will be added to self

        :return: The result of the addition
        '''
        assert len(self.data) == len(sec_obj.data), f'[ERROR] length dimension must match, got {len(self.data)} and {len(sec_obj.data)}'

        res_obj = SimpleTensor(
            data=[self.data[i] + sec_obj.data[i] for i in range(len(self.data))],
            _children=(self, sec_obj),
            _op='+'
        )

        def _backward() -> None:
            '''
            Updating the gradients of the children Lists 
            '''
            for i in range(len(self.data)):
                self.grad[i] += 1.0 * res_obj.grad[i]
                sec_obj.grad[i] += 1.0 * res_obj.grad[i]
        
        res_obj._backward = _backward

        return res_obj
    
    def __neg__(self) -> 'SimpleTensor':
        '''
        Implement the -self operation with List

        :return: The result of the negation
        '''
        res_obj = SimpleTensor(
            data=[-d for d in self.data],
            _children=(self,),
            _op='-'
        )

        def _backward() -> None:
            '''
            Updating the gradients of the children Lists 
            '''
            for i in range(len(self.data)):
                self.grad[i] += -1 * res_obj.grad[i]
        
        res_obj._backward = _backward

        return res_obj

    def __sub__(self, sec_obj: 'SimpleTensor') -> 'SimpleTensor':
        '''
        Implement the __sub__ method for List

        :param sec_obj: The second operand of the subtraction

        :return: The result opf the subtraction
        '''
        return self + (-sec_obj)
    
    def __mul__(self, sec_obj: 'SimpleTensor') -> 'SimpleTensor':
        '''
        Implement the multiplication operator between two Lists

        :param sec_obj: The second operand of the multiplication

        :return: The result of the multiplication
        '''
        assert len(self.data) == len(sec_obj.data), f'[ERROR] length dimension must match, got {len(self.data)} and {len(sec_obj.data)}'

        res_obj = SimpleTensor(
            data=[self.data[i] * sec_obj.data[i] for i in range(len(self.data))],
            _children=(self, sec_obj),
            _op='*'
        )

        def _backward() -> None:
            '''
            Updating the gradients of the children Lists 
            '''
            for i in range(len(self.data)):
                self.grad[i] += sec_obj.data[i] * res_obj.grad[i]
                sec_obj.grad[i] += self.data[i] * res_obj.grad[i]

        res_obj._backward = _backward

        return res_obj
    
    def __pow__(self, exp: Union[int, float]) -> 'SimpleTensor':
        '''
        Implement the self**exp operator for the List

        :parap exp: The exponent of the power operation

        :return: The result of the power operation
        '''
        res_obj = SimpleTensor(
            data = [self.data[i]**exp for i in range(len(self.data))],
            _children=(self, ),
            _op=f'**{exp}'
        )

        def _backward() -> None:
            '''
            Updating the gradients of the children Lists 
            '''
            for i in range(self.data):
                self.grad[i] += exp*(self.data[i]**(exp-1)) * res_obj.grad[i]
            
        res_obj._backward = _backward

        return res_obj

    def __truediv__(self, sec_obj: 'SimpleTensor') -> 'SimpleTensor':
        '''
        Implement the division operation between two Lists

        :param sec_obj: The second operand of the division

        :return: The result of the devision
        '''
        return self * (sec_obj ** -1)

    def exp(self) -> 'SimpleTensor':
        '''
        Implement the e^X operation for the List

        :return: The result of the e^X function
        '''
        res_obj = SimpleTensor(
            data=[math.exp(self.data[i]) for i in range(len(self.data))],
            _children=(self,),
            _op='exp'
        )

        def _backward() -> None:
            '''
            Updating the gradients of the children Lists 
            '''
            for i in range(len(self.data)):
                self.grad[i] += res_obj.data[i] * res_obj.grad[i]
        
        res_obj._backward = _backward

        return res_obj
    
    def tanh(self) -> 'SimpleTensor':
        '''
        Implement the tanh function for Lists

        :return: The result of the tanh function
        '''
        res_obj = SimpleTensor(
            data=[(math.exp(2 * self.data[i]) - 1) / (math.exp(2 * self.data[i]) + 1) for i in range(len(self.data))],
            _children=(self,),
            _op='tanh'
        )

        def _backward() -> None:
            '''
            Updating the gradients of the children Lists 
            '''
            for i in range(len(self.data)):
                self.grad[i] += (1- res_obj.data[i]**2) * res_obj.grad[i]
            
        res_obj._backward = _backward

        return res_obj

    def relu(self) -> 'SimpleTensor':
        '''
        Implement the ReLU function for Lists

        :return: The result of the ReLU function
        '''
        res_obj = SimpleTensor(
            data=[self.data[i] if self.data[i] > 0 else 0 for i in range(len(self.data))],
            _children=(self,),
            _op='relu'
        )

        def _backward() -> None:
            '''
            Updating the gradients of the children Lists 
            '''
            for i in range(len(self.data)):
                self.grad[i] += (self.data[i] > 0) * res_obj.grad[i]
        
        res_obj._backward = _backward

        return res_obj

    def backward(self) -> None:
        '''
        Implement backpropagation for the List class
        '''
        topo = []
        visited = set()
        def _build_topo(v) -> None:
            '''
            Apply topological sorting to the children of a List

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
        self.grad = [1.0 for _ in range(len(self.data))]

        # Apply topological sorted to compute the _backward method of every children
        for n in reversed(topo):
            n._backward()
