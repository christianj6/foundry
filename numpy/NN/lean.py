import numpy as np
np.seterr(all="ignore")


def train(
    layers,
    x,
    y,
    epochs,
    batch_size,
    lr,
):
    x, y = format_data(x, y)
    weights = [np.random.randn(l, layers[i+1]) for i, l in enumerate(layers[:-1])]
    weights.insert(0, np.random.randn(x.shape[1], weights[0].shape[0]))
    weights.append(np.random.randn(layers[-1], y.shape[1]))
    biases = [np.random.randn(1, w.shape[1]) for w in weights]
    data = np.array(list(zip(x, y)))
    for e in range(epochs):
        np.random.shuffle(data)
        batches = np.split(data, data.shape[0] / batch_size)
        for b in batches:
            inp, lab = zip(*b)
            activations, drives = forward(np.array(inp), weights, biases)
            cost = np.array(lab) - activations[-1]
            delta_w = []
            delta_b = []
            for i, d in enumerate(drives[::-1]):
                if i == 0:
                    delta = cost

                else:
                    # Weight after.
                    delta = np.dot(delta, weights[-i].T)

                delta = delta * (1 / (1 + np.exp(-d)) * (1 - 1 / (1 + np.exp(-d))))
                # Activation before.
                delta_w.append(np.dot(activations[-(i+2)].T, delta))
                delta_b.append(delta)

            weights = [w+dw for w, dw in zip(weights, reversed(delta_w))]
            biases = [b+db for b, db in zip(biases, reversed(delta_b))]

        print(e, end=' ')

    return weights, biases


def format_data(x, y):
    onehot = np.eye(np.unique(y).shape[0])
    y = np.array([onehot[a] for a in y])

    return x.reshape((x.shape[0], -1)), y


def forward(
    inp,
    weights,
    biases
):
    activations = [inp]
    drives = []
    for w, b in zip(weights, biases):
        drive = np.dot(activations[-1], w) + b
        drives.append(drive)
        activations.append(1 / (1 + np.exp(-drive)))

    return activations, drives


def evaluate(
    model,
    x,
    y,
    batch_size,
):
    weights, biases = model
    x, y = format_data(x, y)
    data = np.array(list(zip(x, y)))
    batches = np.split(data, data.shape[0] / batch_size)
    acc = []
    for b in batches:
        inp, lab = zip(*b)
        activations, _ = forward(np.array(inp), weights, biases)
        for a, l in zip(activations, lab):
            if np.argmax(l) == np.argmax(a):
                acc.append(1)

            else:
                acc.append(0)

    return sum(acc) / len(acc)
