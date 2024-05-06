from tensors.tensor import Tensor
from tensors.utils import randn, dot


class Module:
    '''
    The base class containing common functionality accross all layers
    '''
    def zero_grad(self) -> None:
        '''
        Reseting layer's gradients
        '''
        for p in self.parameters():
            p.zero_grad()

    def parameters(self) -> Tensor:
        '''
        The tensor containing all the parameters in the layer
        '''
        return Tensor()


class Linear(Module):

    def __init__(self, in_features, out_features):
        self.w = randn(dimensions=(in_features, out_features))
        self.b = randn(dimensions=(out_features,))
    
    def __call__(self, x):
        res = Tensor(dimensions=x.dimensions)

        res = dot(x, self.w)
        print(res.shape(), self.b.shape())

        res += self.b

        return res
