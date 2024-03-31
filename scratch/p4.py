import numpy as np 

np.random.seed(0)

X = [[1, 2, 3, 2.5],
     [2.0, 5.0, -1.0, 2.0],
     [-1.5, 2.7, 3.3, -0.8]]

#X = 3 x 4
#c x l, coluna x linha

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

#layer1 4 X 5
layer1 = Layer_Dense(4,5)
#layer2 5 X 2
layer2 = Layer_Dense(5,2)
layer1.forward(X)
#print(layer1.output)
layer2.forward(layer1.output)
print(layer2.output)

'''
3 X 4 * 4 X 5 = 3 X 5

3 X 5 * 5 X 2 = 3 X 2

'''