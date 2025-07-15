import numpy as np

from npnn.ops import AddOperator, MatMulOperator, MultiplyOperator, PowerOperator, SubtractOperator

class NeuralTensor:
    __slots__ = ('data', 'grad', 'requires_grad', 'operation', 'inputs', 'ctx')
    
    def __init__(self, data: np.ndarray, requires_grad: bool = True):
        self.data = data
        self.grad = np.zeros_like(data) if requires_grad else None
        self.requires_grad = requires_grad
        self.operation = None
        self.inputs = []
        self.ctx = {}
    
    def zero_grad(self):
        if self.grad is not None:
            self.grad.fill(0)
        self.inputs.clear()
        self.ctx.clear()
    
    def backward(self, grad: np.ndarray = None):
        if grad is None:
            grad = np.ones_like(self.data)
        if self.grad is not None:
            if list(self.grad.shape) == list(grad.shape):
                self.grad += grad
            else:
                self.grad += grad.mean(axis=0)
            
        if self.operation:
            # print(f'{grad = }, {self.inputs = }')
            input_grads = self.operation.backward(self.ctx, grad, self.inputs)
            for inp, inp_grad in zip(self.inputs, input_grads):
                if isinstance(inp, NeuralTensor) and inp.requires_grad and inp_grad is not None:
                    inp.backward(inp_grad)

    def __getitem__(self, *args):
        return self.data.__getitem__(*args)
    
    def __sub__(self, other):
        return SubtractOperator.apply(self, other)
    
    def __pow__(self, exponent):
        return PowerOperator.apply(self, exponent)
    
    def __mul__(self, other):
        return MultiplyOperator.apply(self, other)
    
    def __add__(self, other):
        return AddOperator.apply(self, other)
    
    # 在NeuralTensor类中添加矩阵乘法支持
    def __matmul__(self, other):
        return MatMulOperator.apply(self, other)
    
    def __repr__(self):
        return f"NeuralTensor({self.data}, grad={self.grad}, requires_grad={self.requires_grad})"
