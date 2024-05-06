from tensors import Tensor
from tensors.utils import randn, dot, T, flatten
from nn.linear import Linear

def main() -> None:
    '''
    The testing function of the List class
    '''

    ## Backpropagation on a Neuron
    # x1 = Tensor([2.0, 0.0], label="x1")
    # x2 = Tensor([0.0, 2.0], label="x2")
    # w1 = Tensor([-3.0, 1.0], label="w1")
    # w2 = Tensor([1.0, -3.0], label="w2")
    # b = Tensor([6.88137, 6.88137], label="b")

    # x1w1 = x1.dot(w1) ; x1w1.label = "x1*w1"
    # x2w2 = x2.dot(w2) ; x2w2.label = "x2*w2"
    # x1w1_x2w2 = x1w1 + x2w2 ; x1w1_x2w2.label = "x1*w1 + x2*w2"
    # n = x1w1_x2w2 + b ; n.label = "n"

    # # Using as activation function the `tanh`
    # o = n.tanh() ; o.label = 'o'
    # o.backward()

    # x1.zero_grad()

    # print(o)
    # print(x1, x1.grad)
    # print(x2, x2.grad)

    x1 = Tensor([2.0, 1.0])
    w1 = Tensor([[0.0, 2.0, 1.0, 1.0], [1.0, 2.0, 1.0, 2.0]])
    print(x1.shape())
    print(w1.shape())
    # x2 = Tensor([-3.0, 1.0])
    # w2 = Tensor([[0.0, 1.0, 2.0, 1.0], [0.0, 1.0, 2.0, 1.0]])
    # b = Tensor([6.88137, 6.88137, 6.88137, 6.88137])

    x1w1 = x1.dot(w1)
    x1w1.backward()

    print(x1.grad, ' -- ', w1.grad)
    # x2w2 = x2.dot(w2) ; x2w2.label = "x2*w2"
    # x1w1_x2w2 = x1w1 + x2w2 ; x1w1_x2w2.label = "x1*w1 + x2*w2"
    # n = x1w1_x2w2 + b ; n.label = "n"

    # # Using as activation function the `tanh`
    # o = n.tanh() ; o.label = 'o'
    # o.backward()

    # x1.zero_grad()

    # print(o)
    # print(x1, x1.grad)
    # print(x2, x2.grad)

    # t1 = Tensor([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]])
    # t2 = Tensor([[[-1, 0, 1, 2, 3], [2, 3, 4, 1, 1], [5, 6, 7, 9, 1]]])

    # print(t1.shape(), t1)
    # print(t2.shape(), t2)

    # t = t1.dot(t2)
    # t = dot(t1, t2)

    # print(t)
    # print(t.shape())

    # t = Tensor([[2, 1], [2, 1]])

    # model = Linear(t.dimensions[-1], 5)

    # out = model(t)

    # print(out.shape())

    # t1 = Tensor([1, 2, 3])
    # t2 = Tensor([[[[7, 8, 9]]]])
    # t3 = Tensor([1, 2, 3])

    # t = t1 + t2

    # t.backward()

    # print(t1.grad, ' -- ', t2.grad)
    # print(t)
    # print(t.shape())



if __name__ == '__main__':
    main()


# from torch.nn import Linear
# from torch import tensor

# t = tensor([[[2.0, 3.2], [2.2, .33], [2.2, 1.33]]])

# print(t.shape)

# model = Linear(in_features=2, out_features=5)

# for p in model.parameters():
#     print(p)

# out = model(t)

# print(out)

# from itertools import zip_longest

# y = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
# z = [[-1, 2, 5], [0, 3, 6], [1, 4, 7], [2, 1, 9], [3, 1, 1]]

# for zip_element in zip_longest(y, z):
#     print(zip_element)
