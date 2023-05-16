import numpy as np

# y = W * a + b
class Linear():
    def __init__(self, n_in, n_out):
        self.W = np.random.rand(n_in, n_out)
        self.b = np.random.rand(n_out, 1)

    def _forward(self, a):
        self.a = a
        self.z = np.dot(self.W.T, a) + self.b
        return self.z

    def _backward(self, gradient):
        m = gradient.shape[1]
        alpha = 0.05
        
        self.gradient = np.dot(self.W, gradient)
        self.dW = 1./ m * np.sum(np.dot(self.a, gradient.T), axis=1, keepdims=True)
        self.db = 1./ m * np.sum(gradient, axis=1, keepdims=True)

        # gradient descent
        self.W = self.W - alpha * self.dW
        self.b = self.b - alpha * self.db

        return self.gradient
    
    def __call__(self, a):
        return self._forward(a)