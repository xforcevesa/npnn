import numpy as np


class NeuralOperator:
    @classmethod
    def apply(cls, *inputs):
        from npnn.tensor import NeuralTensor
        ctx = {}
        processed_inputs = []
        for inp in inputs:
            if not isinstance(inp, NeuralTensor):
                if isinstance(inp, str):
                    pass
                else:
                    inp = NeuralTensor(np.array(inp), requires_grad=False)
            processed_inputs.append(inp)
        
        output_data = cls.forward(ctx, *[inp.data if isinstance(inp, NeuralTensor) else inp for inp in processed_inputs])
        requires_grad = any(inp.requires_grad for inp in processed_inputs if isinstance(inp, NeuralTensor))
        
        output = NeuralTensor(output_data, requires_grad)
        output.operation = cls
        output.inputs = processed_inputs
        output.ctx = ctx
        return output

class SubtractOperator(NeuralOperator):
    @staticmethod
    def forward(ctx, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return a - b
        
    @staticmethod
    def backward(ctx, grad: np.ndarray, inputs: list) -> list:
        a, b = inputs
        grad_a = grad.copy() if a.requires_grad else None
        grad_b = -grad.copy() if b.requires_grad else None
        return [grad_a, grad_b]
    
class MatMulOperator(NeuralOperator):
    @staticmethod
    def forward(ctx, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        ctx['a_shape'] = a.shape
        ctx['b_shape'] = b.shape
        return np.matmul(a, b)
        
    @staticmethod
    def backward(ctx, grad: np.ndarray, inputs: list) -> list:
        a, b = inputs
        grad_a = np.matmul(grad, b.data.T) if a.requires_grad else None
        grad_b = np.matmul(a.data.T, grad) if b.requires_grad else None
        return [grad_a, grad_b]

class PowerOperator(NeuralOperator):
    @staticmethod
    def forward(ctx, base: np.ndarray, exponent: np.ndarray) -> np.ndarray:
        ctx['base'] = base
        ctx['exponent'] = exponent
        return np.power(base, exponent)
        
    @staticmethod
    def backward(ctx, grad: np.ndarray, inputs: list) -> list:
        base = ctx['base']
        exponent = ctx['exponent']
        base_tensor = inputs[0]
        grad_base = grad * exponent * np.power(base, exponent-1)
        return [grad_base, None]

class MultiplyOperator(NeuralOperator):
    @staticmethod
    def forward(ctx, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        ctx['a'] = a
        ctx['b'] = b
        return a * b
        
    @staticmethod
    def backward(ctx, grad: np.ndarray, inputs: list) -> list:
        a, b = inputs
        grad_a = grad * ctx['b'] if a.requires_grad else None
        grad_b = grad * ctx['a'] if b.requires_grad else None
        return [grad_a, grad_b]

class AddOperator(NeuralOperator):
    @staticmethod
    def forward(ctx, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return a + b
        
    @staticmethod
    def backward(ctx, grad: np.ndarray, inputs: list) -> list:
        a, b = inputs
        grad_a = grad.copy() if a.requires_grad else None
        grad_b = grad.copy() if b.requires_grad else None
        return [grad_a, grad_b]

class ReLUOperator(NeuralOperator):
    @staticmethod
    def forward(ctx, x: np.ndarray) -> np.ndarray:
        ctx['x'] = x
        return np.maximum(0, x)
        
    @staticmethod
    def backward(ctx, grad: np.ndarray, inputs: list) -> list:
        x = ctx['x']
        grad_input = grad * (x > 0)
        return [grad_input]

class SoftmaxOperator(NeuralOperator):
    @staticmethod
    def forward(ctx, x: np.ndarray) -> np.ndarray:
        # 数值稳定性的softmax
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        softmax = exp_x / np.sum(exp_x, axis=-1, keepdims=True)
        ctx['softmax'] = softmax
        return softmax
        
    @staticmethod
    def backward(ctx, grad: np.ndarray, inputs: list) -> list:
        softmax = ctx['softmax']
        # 使用雅可比矩阵计算梯度
        s = softmax.reshape(-1, 1)
        jacobian = np.diagflat(s) - np.dot(s, s.T)
        grad_input = np.dot(grad, jacobian)
        return [grad_input.reshape(softmax.shape)]

class CrossEntropyLossOperator(NeuralOperator):
    @staticmethod
    def forward(ctx, logits: np.ndarray, labels: np.ndarray) -> np.ndarray:
        # 数值稳定的softmax交叉熵
        max_logits = np.max(logits, axis=1, keepdims=True)
        stable_logits = logits - max_logits
        exp_logits = np.exp(stable_logits)
        softmax = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        
        # 计算交叉熵损失
        batch_size = logits.shape[0]
        correct_log_probs = -np.log(softmax[range(batch_size), labels] + 1e-8)
        loss = np.sum(correct_log_probs) / batch_size
        
        ctx['softmax'] = softmax
        ctx['labels'] = labels
        return np.array(loss)
        
    @staticmethod
    def backward(ctx, grad: np.ndarray, inputs: list) -> list:
        softmax = ctx['softmax']
        labels = ctx['labels']
        batch_size = softmax.shape[0]
        
        # 计算梯度
        grad_logits = softmax.copy()
        grad_logits[range(batch_size), labels] -= 1
        grad_logits /= batch_size
        
        return [grad_logits, None]  # 标签不需要梯度
