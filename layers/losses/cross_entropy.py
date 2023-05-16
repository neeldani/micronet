import numpy as np

# a = max(z, o)
class CrossEntropy():
    def _forward(self, y_pred, y_true):
        self.y_true = y_true
        self.y_pred = y_pred
        m = y_pred.shape[1]

        self.loss = -1./m * np.sum(np.multiply(self.y_true, np.log(self.y_pred)))
        return np.squeeze(self.loss)

    def _backward(self):
        self.gradient = - self.y_true/self.y_pred
        return self.gradient
    
    def __call__(self, y_pred, y_true):
        return self._forward(y_pred, y_true)