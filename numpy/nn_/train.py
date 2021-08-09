from keras.datasets import mnist
from model import Model


(x_train, y_train), (x_test, y_test) = mnist.load_data()
nn = Model([30, 60, 60, 30])
nn.fit(x_train, y_train, epochs=50, batch_size=40, lr=0.003)
print(nn.eval(x_test, y_test))
