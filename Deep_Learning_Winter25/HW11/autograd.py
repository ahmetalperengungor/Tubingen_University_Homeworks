import numpy as np
from graphviz import Digraph


class Tensor:
    all_tensors = []

    def __init__(self, data, inputs=()):
        """ data: value of this tensor, inputs: list of input tensors """
        self.id = len(self.all_tensors)
        self.data = data
        self.inputs = inputs
        self.grad = np.zeros_like(data)
        self.all_tensors.append(self)

        self.forward_usages = 0
        self.backward_usages = 0
        for input_ in self.inputs:
            input_.forward_usages += 1

    def prev(self) -> list:
        return self.inputs

    def print(self):
        print('%d)' % self.id, self.__class__.__name__)
        print('data\n', self.data)
        print('grad\n', self.grad)
        print('inputs', [p.id for p in self.prev()])

    def graph(self, graph=None):
        """ returns a graphviz graph of all dependencies """
        if graph is None:
            graph = Digraph(format='pdf', node_attr=dict(style='filled', shape='rect'))
        graph.node(str(self.id), label=self.name(), fillcolor='white' if len(self.inputs) == 0 else 'lightblue')
        for ins in self.inputs:
            ins.graph(graph)
            graph.edge(str(ins.id), str(self.id))
        return graph

    def name(self) -> str:
        return '%d: %s' % (self.id, self.__class__.__name__)
        
    @classmethod
    def reset_tensors(cls):
        cls.all_tensors = []

    @classmethod
    def reset_grads(cls):
        for tensor in cls.all_tensors:
            tensor.grad.fill(0)

    def backward(self, start=True):
        """ Recursively back-propagate gradients """
                # TODO (a)
        # if we start backpropagating here, each value has a gradient of one
        # call the _assign_grads() method once for all Tensors that use this one as input have assigned their gradients
        #   hint: self.forward_usages, self.backward_usages to track that
        #   then also recursively continue with every input of this tensor
        if start:
            self.grad.fill(1.0)
        
        self.backward_usages += 1
        
        if start or self.backward_usages == self.forward_usages:
            self._assign_grads()
            for input_ in self.inputs:
                input_.backward(start=False)

    def _assign_grads(self):
        """ Compute and assign gradients for each input """
        raise NotImplementedError


class Variable(Tensor):
    def __init__(self, name, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._name = name

    def _assign_grads(self):
        pass

    def name(self) -> str:
        return '%d: %s (%s)' % (self.id, self._name, self.__class__.__name__)


class Neg(Tensor):
    def __init__(self, a: Tensor):
        result = -a.data
        super().__init__(result, [a])
        self.a = a

    def _assign_grads(self):
        # add gradients to inputs of this graph node
        self.a.grad -= self.grad


class Sqrt(Tensor):
    def __init__(self, a: Tensor):
        result = np.sqrt(a.data)
        super().__init__(result, [a])
        self.a = a

    def _assign_grads(self):
        # add gradients to inputs of this graph node
        self.a.grad += self.grad / (2 * np.sqrt(self.a.data))


class ReduceMean(Tensor):
    def __init__(self, a: Tensor):
        result = np.mean(a.data)
        super().__init__(result, [a])
        self.a = a

    def _assign_grads(self):
        # TODO (b)
        self.a.grad += self.grad / self.a.data.size


class Add(Tensor):
    def __init__(self, *tensors: [Tensor]):
        result = sum([v.data for v in tensors])
        super().__init__(result, tensors)

    def _assign_grads(self):
        # TODO (b)
        for tensor in self.inputs:
            tensor.grad += self.grad


class Mul(Tensor):
    def __init__(self, a: Tensor, b: Tensor):
        result = a.data * b.data
        super().__init__(result, [a, b])
        self.a = a
        self.b = b

    def _assign_grads(self):
        # TODO (b)
        self.a.grad += self.grad * self.b.data
        self.b.grad += self.grad * self.a.data


class MatMul(Tensor):
    def __init__(self, a: Tensor, b: Tensor):
        result = np.matmul(a.data, b.data)
        super().__init__(result, [a, b])
        self.a = a
        self.b = b

    def _assign_grads(self):
        self.a.grad += np.matmul(self.grad, self.b.data.T)
        self.b.grad += np.matmul(self.a.data.T, self.grad)


class ReLU(Tensor):
    def __init__(self, a: Tensor):
        result = (a.data > 0) * a.data
        super().__init__(result, [a])
        self.a = a

    def _assign_grads(self):
        self.a.grad += self.grad * (self.a.data > 0)


class Sigmoid(Tensor):
    def __init__(self, a: Tensor):
        result = 1 / (1 + np.exp(-a.data))
        super().__init__(result, [a])
        self.a = a

    def _assign_grads(self):
        # TODO (b)
        self.a.grad += self.grad * self.data * (1 - self.data)


def mse(a: Tensor, b: Tensor) -> Tensor:
    # TODO (c)
    diff = Add(a, Neg(b))
    sq = Mul(diff, diff)
    return ReduceMean(sq)


def main():
    # fixed input
    v0 = Variable('A', np.array([[-5.0, -2.0, -1.0], [-5.0, -1.0, -3.0]]))

    # example computation graph, built on the fly
    vx = Neg(v0)
    vx = Sqrt(vx)

    # start backpropagating
    vx.backward()

    # print gradients
    for v_ in Tensor.all_tensors:
        # if not isinstance(v_, Variable):
        #     continue
        print('-'*50)
        v_.print()

    # show the computation graph
    vx.graph().render('/tmp/autograd/graph', view=True)

    # reset all tensors, since the graph is only built for a single forward+backward pass
    Tensor.reset_tensors()


if __name__ == '__main__':
    main()
