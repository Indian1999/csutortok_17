import numpy as np

class NeuralNetwork():
    def __init__(self):
        self.weights = np.random.random(size = (4, 1)) * 2 - 1    # [-1, 1) generálunk 4*1 számot
        self.bias = -1
        
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_deriv(self, x):
        return x * (1-x)

    def predict(self, inputs):
        return self.sigmoid(np.dot(inputs, self.weights) + self.bias) 
    
    def train(self, train_x, train_y, epochs):
        history = []
        for i in range(epochs):
            output = self.predict(train_x) # output: 8x1-es mtx
            error = train_y - output
            adjust = np.dot(train_x.T, error * self.sigmoid_deriv(output))
            self.weights += adjust
            print(f"{i+1}. epoch:")
            print(f"Error: {error.flatten()}")
            print(f"Adjust: {adjust.flatten()}")
            print(f"Weights: {self.weights.flatten()}")
            print(f"Loss: {error.mean()}")
            history.append(error.mean())
        return history

model = NeuralNetwork()

train_x = np.array([
        [1,0,1,0],
        [0,1,0,1],
        [1,1,1,1],
        [0,0,0,0],
        [1,1,0,0],
        [0,1,1,0],
        [0,1,0,0],
        [1,0,0,1]
    ]) # 8x4-es mátrix
train_y = np.array([item[0] for item in train_x]) # 1 dimenziós 8 elemű tömb
train_y = train_y.reshape((-1, 1)) # 8x1-es mátrixot csinálunk belőle

print("Weights before training:", model.weights.flatten())
history = model.train(train_x, train_y, 100)
print("Weights after training:", model.weights.flatten())

import matplotlib.pyplot as plt

#plt.plot(history)
#plt.show()