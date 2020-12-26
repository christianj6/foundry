from keras.datasets import mnist
from lean import train, evaluate
from model import Model


(x_train, y_train), (x_test, y_test) = mnist.load_data()
nn = train(
        [30, 60, 60, 30],
        x_train,
        y_train,
        epochs=10,
        batch_size=40,
        lr=0.003
    )
print(evaluate(
        nn,
        x_test,
        y_test,
        40,
    ))
# nn = Model([30, 60, 60, 30])
# nn.fit(x_train, y_train, epochs=50, batch_size=40, lr=0.003)
# print()
# print(nn.eval(x_test, y_test))
