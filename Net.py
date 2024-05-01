import numpy as np
import time
from PIL import Image
from matplotlib import pyplot as plt

from image_process import get_a_sample, shuffle


def sigmoid(x, derivative=False):
    if derivative:
        return x * (1 - x)
    return 1 / (1 + np.exp(-x))


def softmax(x, derivative=False):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


def cross_entropy_loss(y, y_hat, derivative=False):
    if derivative:
        return y_hat - y
    return -np.mean(y * np.log(y_hat + 1e-15))


class Layer:
    def __init__(self, input_size, output_size, activation):
        self.output = None
        self.input = None
        self.input_size = input_size
        self.output_size = output_size
        self.activation = activation
        # self.weights = np.random.uniform(0.02, 0.1, size=(input_size, output_size))
        self.weights = np.random.randn(input_size, output_size) * np.sqrt(
            2.0 / input_size
        )
        self.biases = np.random.uniform(-0.05, 0, size=(1, output_size))

    def forward(self, x):
        self.input = x
        self.output = np.dot(x, self.weights) + self.biases
        self.output = self.activation(self.output)
        return self.output

    def backward(self, delta, learning_rate):
        if self.activation == softmax:
            # softmax_derivative = np.diag(self.output) - np.dot(
            #     self.output.T, self.output
            # )
            # delta = np.dot(delta, softmax_derivative)
            delta = delta

        self.weights -= learning_rate * np.dot(self.input.T, delta)
        self.biases -= learning_rate * np.sum(delta, axis=0)
        return np.dot(delta, self.weights.T)


class NeuralNetwork:
    def __init__(self, input_size, hidden_layer_sizes, output_size, learning_rate):
        self.input_size = input_size
        self.hidden_layer_sizes = hidden_layer_sizes
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.loss_func = cross_entropy_loss

        self.layers = []
        self.layers.append(Layer(input_size, hidden_layer_sizes[0], sigmoid))
        for i in range(1, len(hidden_layer_sizes)):
            self.layers.append(
                Layer(hidden_layer_sizes[i - 1], hidden_layer_sizes[i], sigmoid)
            )
        self.layers.append(Layer(hidden_layer_sizes[-1], output_size, softmax))

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, y):
        delta = self.loss_func(y, self.layers[-1].output, derivative=True)
        for layer in reversed(self.layers):
            delta = layer.backward(delta, self.learning_rate)

    def train(self, train_data, labels):
        cnt = 0
        for x, y in zip(train_data, labels):
            self.forward(x)
            self.backward(y)

    def predict(self, x):
        return self.forward(x)


learning_rate = 0.0005
nn = NeuralNetwork(28 * 28, [20], 12, learning_rate)
start = time.time()
times = 5
z = ["博", "学", "笃", "志", "切", "问", "近", "思", "自", "由", "无", "用"]
accuracy = []
x = []
y = []
get_a_sample(x, y)

shuffle(x, y)
for i in range(times):
    accuracy.append(0)
    nn.train(x, y)
    for j in range(len(x)):
        accuracy[i] += (np.argmax(nn.predict(x[j])) == np.argmax(y[j])) / len(x)

print(accuracy)
end = time.time()
print(end - start)

# Plotting the accuracy trend
xx = range(len(accuracy))
plt.plot(xx, accuracy, label=accuracy[-1])
plt.legend()
plt.title("TREND")
plt.show()

user_input = input("输入两个值来定位图片的位置，或者输入exit退出")
while user_input != "exit":
    user_input = user_input.split(" ")
    address = "train/" + user_input[0] + "/" + user_input[1] + ".bmp"
    image = Image.open(address)
    image = image.resize((28, 28)).convert("L")
    image_array = np.array(image)
    image_array = np.ones([28, 28], dtype=int) * 256 - image_array
    # print(image_array)
    plt.imshow(image_array, cmap="gray")
    plt.colorbar()  # 显示颜色条
    plt.show()
    vector_1x784 = image_array.flatten().reshape(1, 784)  # Reshaping to (1, 784)
    prediction = nn.predict(vector_1x784)
    predicted_index = np.argmax(prediction)
    print(
        z[predicted_index], prediction[0][predicted_index]
    )  # Accessing the predicted value correctly
    for i in range(12):
        if i == predicted_index:
            continue
        print(z[i], prediction[0][i])  # Accessing other predictions correctly
    user_input = input("输入两个值来定位图片的位置，或者输入exit退出")
