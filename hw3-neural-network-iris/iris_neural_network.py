import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class NeuralNetworkIris:
    def __init__(self, input_dim=4, hidden_dim=12, output_dim=3, learning_rate=0.05, seed=42):
        np.random.seed(seed)

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.learning_rate = learning_rate

        limit1 = np.sqrt(6 / (input_dim + hidden_dim))
        limit2 = np.sqrt(6 / (hidden_dim + output_dim))

        self.W1 = np.random.uniform(-limit1, limit1, (input_dim, hidden_dim))
        self.b1 = np.zeros((1, hidden_dim))

        self.W2 = np.random.uniform(-limit2, limit2, (hidden_dim, output_dim))
        self.b2 = np.zeros((1, output_dim))

    def tanh(self, z):
        return np.tanh(z)

    def tanh_derivative(self, a):
        return 1 - a ** 2

    def softmax(self, z):
        z_shifted = z - np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(z_shifted)
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.tanh(self.z1)

        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.y_hat = self.softmax(self.z2)

        return self.y_hat

    def cross_entropy_loss(self, y_true, y_pred):
        epsilon = 1e-12
        loss = -np.mean(np.sum(y_true * np.log(y_pred + epsilon), axis=1))
        return loss

    def backward(self, X, y_true):
        m = X.shape[0]

        dz2 = (self.y_hat - y_true) / m

        dW2 = np.dot(self.a1.T, dz2)
        db2 = np.sum(dz2, axis=0, keepdims=True)

        da1 = np.dot(dz2, self.W2.T)
        dz1 = da1 * self.tanh_derivative(self.a1)

        dW1 = np.dot(X.T, dz1)
        db1 = np.sum(dz1, axis=0, keepdims=True)

        self.W2 = self.W2 - self.learning_rate * dW2
        self.b2 = self.b2 - self.learning_rate * db2

        self.W1 = self.W1 - self.learning_rate * dW1
        self.b1 = self.b1 - self.learning_rate * db1

    def train(self, X_train, y_train, epochs=1500, print_every=300):
        losses = []

        for epoch in range(1, epochs + 1):
            y_pred = self.forward(X_train)
            loss = self.cross_entropy_loss(y_train, y_pred)
            losses.append(loss)

            self.backward(X_train, y_train)

            if epoch == 1 or epoch % print_every == 0:
                print(f"Epoch {epoch:4d} | Loss: {loss:.4f}")

        return losses

    def predict(self, X):
        y_pred = self.forward(X)
        return np.argmax(y_pred, axis=1)

    def accuracy(self, X, y_true):
        y_pred = self.predict(X)
        return np.mean(y_pred == y_true)


iris = load_iris()
X = iris.data
y = iris.target

y_one_hot = np.eye(3)[y]

X_train, X_test, y_train, y_test, y_train_one_hot, y_test_one_hot = train_test_split(
    X,
    y,
    y_one_hot,
    test_size=0.2,
    random_state=42,
    stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = NeuralNetworkIris(
    input_dim=4,
    hidden_dim=12,
    output_dim=3,
    learning_rate=0.05,
    seed=42
)

losses = model.train(
    X_train,
    y_train_one_hot,
    epochs=1500,
    print_every=300
)

test_accuracy = model.accuracy(X_test, y_test)

print("Test Accuracy:", test_accuracy)