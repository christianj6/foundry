from keras.datasets import mnist
import matplotlib.pyplot as plt
from model import Model


(x_train, y_train), (x_test, y_test) = mnist.load_data()
nn = Model([100])
nn.fit(x_train, y_train, epochs=5, batch_size=10, lr=0.3)
print()
print(nn.eval(x_test, y_test))

# ----
# TODO: Layer object so we can use different activation functions.
# TODO: Dropout
