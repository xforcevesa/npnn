import numpy as np
from npnn.functional import relu, softmax
from npnn.tensor import NeuralTensor

class LinearLayer:
    def __init__(self, input_size, output_size):
        # 使用Xavier初始化权重
        self.weights = NeuralTensor(
            np.random.randn(input_size, output_size) * np.sqrt(2. / (input_size + output_size)),
            requires_grad=True
        )
        self.bias = NeuralTensor(np.zeros((1, output_size)), requires_grad=True)
    
    def __call__(self, x):
        return (x @ self.weights) + self.bias
    
    def parameters(self):
        return [self.weights, self.bias]
