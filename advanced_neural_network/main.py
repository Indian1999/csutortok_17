from SmartGrass.network import Network
from SmartGrass.dense import Dense
from SmartGrass.activation import Activation
from SmartGrass.activations import tanh, tanh_deriv
from SmartGrass.losses import mse, mse_deriv
import numpy as np

model = Network()
model.add(Dense(5, 6))
model.add(Activation(tanh, tanh_deriv))
model.add(Dense(6, 1))
model.add(Activation(tanh, tanh_deriv))

model.use_loss(mse, mse_deriv)

# 5 bit lesz a bemenet
# ha az 1. és 4. bit 1, akkor a kimenet 1, különben 0

train_len = 25

train_x = np.random.randint(0, 2, size = (train_len, 1, 5))
train_y = []
for item in train_x:
    if item[0][0] == 1 and item[0][4] == 1:
        train_y.append(1)
    else:
        train_y.append(0)
train_y = np.array(train_y).reshape(train_len, 1, 1)

model.fit(train_x, train_y, epochs=100, learning_rate=0.1)

while True:
    bits = input("Adj meg 5 bitet: ")
    bits = np.array(list(bits)).astype("float32")
    bits = bits.reshape(1, 1, 5)
    result = model.predict(bits)[0][0][0]
    print(round(result))

