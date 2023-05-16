import numpy as np

# a = sigmoid(z)
class Sigmoid():
    def _forward(self, z):
        self.z = z
        log_probs = np.exp(z)
        self.a = log_probs / np.sum(log_probs, axis=0, keepdims=True)
        return self.a

    def _backward(self, gradient):
        self.gradient = self.z * (1 - self.z)
        return self.gradient
    
    def __call__(self, z):
        return self._forward(z)