class NeuralOptimizer:
    def __init__(self, params: list, lr: float):
        self.params = [p for p in params if p.requires_grad]
        self.lr = lr
        
    def zero_grad(self):
        for param in self.params:
            param.zero_grad()
            
    def step(self):
        for param in self.params:
            if param.grad is not None:
                param.data -= self.lr * param.grad
