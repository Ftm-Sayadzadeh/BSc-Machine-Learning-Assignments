# Iris Classification — Neural Network from Scratch

A fully connected neural network written with **NumPy only** (no PyTorch, TensorFlow or Keras) that
predicts the Iris species from four flower measurements. The forward pass, the softmax output, the
cross-entropy loss, **backpropagation** and the gradient-descent update are all implemented by hand.

## Data

The classic Iris dataset (150 flowers, 3 species with 50 samples each: *setosa*, *versicolor*,
*virginica*) loaded with `sklearn.datasets.load_iris()`. Inputs are sepal length/width and petal
length/width.

## Approach

1. **Split** 80 % / 20 % into train / test (120 / 30 samples), **stratified** so every species has 40 training and 10 test samples; seed 42.
2. **Standardize** the features with the mean and standard deviation of the **training set only**.
3. **Architecture** — `4 → 12 → 3`:

   | Layer | Computation |
   |---|---|
   | Hidden (12 neurons) | `a1 = tanh(X·W1 + b1)` |
   | Output (3 neurons) | `ŷ = softmax(a1·W2 + b2)` |

   Weights use Glorot/Xavier uniform initialization; biases start at zero.
4. **Loss:** categorical cross-entropy on one-hot labels (with a small epsilon inside the `log`).
5. **Backpropagation:** because softmax and cross-entropy are combined, the gradient at the output is simply `(ŷ − y) / m`; it is then propagated back through `W2` and the `tanh` derivative `1 − a1²` to get the gradients of `W1` and `b1`.
6. **Training:** full-batch gradient descent, learning rate 0.05, 1,500 epochs.

The `NeuralNetworkIris` class contains all the pieces: `forward`, `cross_entropy_loss`, `backward` (which also applies the update), `train`, `predict` and `accuracy`. The softmax subtracts the row maximum before exponentiating for numerical stability.

## Results

Training loss:

| Epoch | 1 | 300 | 600 | 900 | 1200 | 1500 |
|---|---|---|---|---|---|---|
| Loss | 2.6763 | 0.1444 | 0.0837 | 0.0655 | 0.0574 | 0.0527 |

**Test accuracy: 96.7 %** (29 of 30 flowers).

```
Confusion matrix (rows = true, columns = predicted)
              setosa  versicolor  virginica
setosa            10           0          0
versicolor         0           9          1
virginica          0           0         10
```

- *Setosa* is separated perfectly, as expected — it is the one species that is linearly separable from the others.
- The only error is a *versicolor* predicted as *virginica*; these two species overlap in feature space, so this is where a small network makes its mistakes.
- The loss falls quickly in the first few hundred epochs and then keeps decreasing slowly, so the model is still (very slightly) improving at epoch 1,500.
- With only 30 test samples, one mistake changes the accuracy by 3.3 points, so the number should be read as "very good" rather than as a precise estimate.
- The hand-written backpropagation was also checked separately against a finite-difference (numerical) gradient: the maximum relative error was about 1e-8. That check is not part of the script.

## Run

```bash
python iris_neural_network.py
```

Prints the loss every 300 epochs and the final test accuracy. The loss history is kept in the `losses` list if you want to plot it.
