import numpy as np

# a = max(z, o)
class Relu():
    def _forward(self, z):
        self.z = z
        self.a = np.maximum(z, 0)
        return self.a

    def _backward(self, gradient):
        self.gradient = self.a
        self.gradient[self.gradient > 0] = 1
        return np.multiply(self.gradient, gradient)
    
    def __call__(self, z):
        return self._forward(z)