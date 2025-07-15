from npnn.ops import *
from npnn.tensor import NeuralTensor


def relu(x):
    return ReLUOperator.apply(x)

def softmax(x):
    return SoftmaxOperator.apply(x)

def cross_entropy_loss(logits, labels):
    return CrossEntropyLossOperator.apply(logits, labels)

def mean_squared_error(y_pred: NeuralTensor, y_true: NeuralTensor) -> NeuralTensor:
    """计算均方误差损失函数[6,8](@ref)"""
    return (y_pred - y_true) ** 2
