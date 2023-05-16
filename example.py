from models.sequential import Sequential
from layers.linear import Linear
from layers.activations.relu import Relu
from layers.activations.softmax import Softmax
from layers.losses.cross_entropy import CrossEntropy

from sklearn import datasets
from sklearn.preprocessing import OneHotEncoder

# preprocess iris
iris = datasets.load_iris()
X = iris.data.T
Y = iris.target.reshape(150, 1)
one_hot = OneHotEncoder()
Y = one_hot.fit_transform(Y.reshape(-1, 1)).todense().T

model = Sequential(
    [
        Linear(X.shape[0], 5), 
        Relu(),
        Linear(5, Y.shape[0]),
        Softmax(),
    ],
    X = X,
    Y = Y,
    batch_size = 32,
    loss_function = CrossEntropy()
)

model.train(epochs=250, alpha=0.01)