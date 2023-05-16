import math
import numpy as np

class Sequential:
    def __init__(self, model, X, Y, batch_size, loss_function):
        self.X = X
        self.Y = Y
        self.model = model
        self.batch_size = batch_size
        self.loss_function = loss_function

    def train(self, epochs, alpha):
        for epoch in range(epochs):
            cost = 0

            for x, y_true in self.get_batches(self.X, self.Y, self.batch_size):
                y_pred = self.predict(x)
                cost += self.loss_function(y_pred, y_true)
                self.update_weights()
            
            if epoch % (epochs * 0.05) == 0:
                print(f"epoch: {epoch+1} cost: {cost:.6f}")
    
    def get_batches(self, X, Y, batch_size):
        num_samples = X.shape[1]
        num_batches = math.ceil(num_samples / batch_size)
    
        for i in range(0, num_batches):
            start_index = i*batch_size
            end_index = min((i + 1) * batch_size, num_samples)
            yield np.array(X[:, start_index:end_index]), np.array(Y[:, start_index:end_index])
    
    def predict(self, x_in):
        for layer in self.model:
            x_in = layer(x_in)
        return x_in
    
    def update_weights(self):
        gradient = self.loss_function._backward()
        for layer in reversed(self.model):
            gradient = layer._backward(gradient)