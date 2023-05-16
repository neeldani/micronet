import numpy as np

# a = sigmoid(z)
class Softmax():
    def _forward(self, z):
        self.z = z
        log_probs = np.exp(z)
        self.a = log_probs / np.sum(log_probs, axis=0, keepdims=True)
        return self.a

    def _backward(self, gradient):
        a = self.a
        J = np.einsum('ij,kj->jik', a, a)
        I = np.eye(a.shape[0])
        J = np.einsum('ij,ik->jik', a, I) - J
        self.gradient = np.einsum('kij,jk->ik', J, gradient)
        return self.gradient
    
    def __call__(self, z):
        return self._forward(z)