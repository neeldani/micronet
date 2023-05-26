# micronet
Micronet is a library which gives you the ability to architect any permutations of sequential neural nets. It uses `numpy` only.

```python
# Example to build your own model

model = Sequential(
    [
        Linear(input_size, hidden_size_1), 
        Relu(),
        Linear(hidden_size_1, hidden_size_2),
        Relu(),
        Linear(hidden_size_2, output_size),
        Softmax(),
    ],
    X = X,
    Y = Y,
    batch_size = 128,
    loss_function = CrossEntropy()
)

model.train(epochs, alpha)
```
