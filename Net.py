import numpy as np
import time
from PIL import Image
from matplotlib import pyplot as plt
from PJA.BP2.NeuralNetwork import NeuralNetwork
from PJA.BP2.image_process import get_a_sample


def tanh(x, derivative=False):
    if derivative:
        return 1 - np.square(x)
    return np.tanh(x)

def sigmoid(x, derivative=False):
    if derivative:
        return x * (1 - x)
    return 1 / (1 + np.exp(-x))

def softmax(x, derivative=False):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def relu(x, derivative=False):
    if derivative:
        return (x > 0).astype(int)
    return np.maximum(0, x)

def cross_entropy_loss(y, y_hat, derivative=False):
    if derivative:
        return y_hat - y
    return -np.mean(y * np.log(y_hat + 1e-15) + (1 - y) * np.log(1 - y_hat + 1e-15))

class Layer:
    def __init__(self, input_size, output_size, activation):
        self.output = None
        self.input = None
        self.input_size = input_size
        self.output_size = output_size
        self.activation = activation
        self.weights = np.random.uniform(0.02, 0.1, size=(input_size, output_size))
        self.biases = np.random.uniform(-0.05, 0, size=(1, output_size))

    def forward(self, x):
        self.input = x
        self.output = np.dot(x, self.weights) + self.biases
        self.output = self.activation(self.output)
        return self.output

    def backward(self, delta, learning_rate):
        if self.activation == softmax:
            softmax_derivative = np.diag(self.output) - np.dot(self.output.T, self.output)
            delta = np.dot(delta, softmax_derivative)
        else:
            delta = delta * self.activation(self.output, derivative=True)
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
        print(delta)
        for layer in reversed(self.layers):
            delta = layer.backward(delta, self.learning_rate)

    def train(self, train_data, labels):
        for x,y in zip(train_data,labels):
            self.forward(x)
            self.backward(y)

    def predict(self, x):
        return self.forward(x)


learning_rate = 0.005
nn = NeuralNetwork(784, [250], 12, learning_rate)
start = time.time()
times = 1
z = ['博', '学', '笃', '志', '切', '问', '近', '思', '自', '由', '无', '用']
accuracy = []
x = []
y = []
get_a_sample(x, y)
for i in range(times):
    nn.train(x, y)
    predict = nn.predict(x)
    accuracy.append(np.sum(np.argmax(predict) == np.argmax(y))/7200)

end = time.time()
print(end-start)

# Plotting the accuracy trend
xx = range(len(accuracy))
plt.plot(xx, accuracy, label=accuracy[-1])
plt.legend()
plt.title('TREND')
plt.show()

user_input = input('输入两个值来定位图片的位置，或者输入exit退出')
while user_input != 'exit':
    user_input = user_input.split(' ')
    address = 'train/'+user_input[0]+'/'+user_input[1]+'.bmp'
    image = Image.open(address)
    image = image.resize((28, 28)).convert('L')
    image_array = np.array(image)
    vector_1x784 = image_array.flatten().reshape(1, 784)  # Reshaping to (1, 784)
    prediction = nn.predict(vector_1x784)
    predicted_index = np.argmax(prediction)
    print(z[predicted_index], prediction[0][predicted_index])  # Accessing the predicted value correctly
    for i in range(12):
        if i == predicted_index:
            continue
        print(z[i], prediction[0][i])  # Accessing other predictions correctly
    user_input = input('输入两个值来定位图片的位置，或者输入exit退出')
