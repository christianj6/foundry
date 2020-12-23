import numpy as np
from tqdm import tqdm
import random
import matplotlib.pyplot as plt


class Model():
    '''
    Basic, configurable feed-forward model.
    '''
    def __init__(self, layers:list=[2, 2]):
        '''
        Initialize network from given
        parameters

        Parameters
        ----------
            layers : list
                List of integers specifying the
                number of nodes in each layer.
        '''
        # Initialize layer shapes.
        self.layers = layers
        self.n_layers = len(layers)
        # Just a list of vectors of appropriate size
        # to each layer's weight matrix.
        self.biases = []
        self.weights = []

    def forward(self, batch):
        '''
        Pass a batch of inputs forward
        through the network.

        Parameters
        ---------
            batch : np.array
                Input batch.

        Returns
        ---------
            activations : list
                List of activations,
                nd.arrays, from each
                layer of the network.
        '''
        # We need to keep the input so we can
        # compute backprop later.
        activations = [np.array(batch)]
        drives = []
        activation = batch
        for w, b in zip(self.weights, self.biases):
            drive = np.dot(activation, w) + b
            drives.append(drive)
            activation = self.sigma(drive)
            activations.append(activation)

        return activations, drives

    @staticmethod
    def sigma(drive):
        '''
        Return activation from
        a given drive.

        Paramters
        ---------
            drive : np.array
                Computes activation
                values using the
                written activation
                function.

        Returns
        ---------
            activation : np.array
                Activation values.
        '''
        # Avoid np overflow.
        drive = np.clip(drive, -500, 500)
        return 1 / (1 + np.exp(-drive))

    def sigma_prime(self, drive):
        '''
        Derivative of sigma activation
        function.

        Paramters
        ---------
            drive : np.array
                Derivative activation
                value given an incoming drive.
        '''
        return self.sigma(drive) * (1 - self.sigma(drive))

    def back(self, activations, drives, labels):
        '''
        Pass updates back
        through the network.

        Parameters
        ---------
            activations : list
                List of activation
                arrays from each layer.
            drives : list
                List of drives from each
                layer.
            labels : np.array
                Ground truth for
                evaluating loss.

        Returns
        ---------
            delta_w : np.array
                Update to weights.
            delta_b : np.array
                Update to biases.
            loss : float
                Update on loss for batch.
        '''
        # Instantiate updates.
        delta_w = []
        delta_b = []
        # Get element-wise loss.
        cost_matrix = activations[-1] - labels
        # Get avg loss.
        loss = np.average(cost_matrix)

        # Flip the lists around so we can easily walk back.
        activations_reversed = [a for a in activations[::-1]]
        drives_reversed = [d for d in drives[::-1]]
        weights_reversed = [w for w in self.weights[::-1]]

        # for i in range(len(self.weights)):
        #     print(self.weights[i].shape)
        # print()

        # Get delta for each layer.
        for i, d in enumerate(drives_reversed):
            # print(i)
            if i == 0:
                # Penultimate layer. Instantiate delta.
                delta = cost_matrix * self.sigma_prime(d)

            else:
                # Other layers, infer delta via backprop.
                delta = np.dot(delta, weights_reversed[i-1].T) * self.sigma_prime(d)

            # print(delta.shape)

            db = delta
            # print(activations[-i+2])
            # print(activations_reversed[i].T.shape)
            # print('--')
            dw = np.dot(activations_reversed[i+1].T, delta)
            # print(dw.shape)
            # Apply the delta to the update matrix.
            delta_w.append(dw)
            delta_b.append(db)

        # exit()

        return reversed(delta_w), reversed(delta_b), loss

    def fit(
        self,
        x,
        y,
        epochs:int=10,
        batch_size:int=100,
        lr:float=0.01
    ):
        '''
        Fits the network to a set of
        training data.

        Parameters
        ---------
            x : np.array
                Raw input array.
            y : np.array
                Output labels.
            epochs : int
                Number of training epochs.
            batch_size : int
                Number of training
                samples in each batch.
            lr : float
                Learning rate.
        '''
        # Store batch size for eval.
        self.batch_size = batch_size
        # Flatten all input arrays.
        X = x.reshape(x.shape[0], -1)
        # Add appropriate layers based on input shape.
        self.weights.extend([np.random.randn(X.shape[1], s) \
                                if i == 0 else (np.random.randn(self.layers[i-1], s)) \
                                for i, s in enumerate(self.layers)])
        # Add biases for the weights.
        self.biases.extend([np.random.randn(1, w.shape[1]) \
                                for w in self.weights])
        # One-hot encoding.
        self.labels = np.eye(np.unique(y).shape[0])
        y = [self.labels[value] for value in y]
        # Add output layer.
        self.weights.append(np.random.randn(self.weights[-1].shape[1], self.labels.shape[0]))
        self.biases.append(np.random.randn(1, self.labels.shape[0]))
        # Organize data for training.
        data = list(zip(X, y))
        n = len(data)
        # Training loop.
        losses = []
        # outer = tqdm(total=epochs, desc='Epoch', position=0)
        for i in range(epochs):
            # Shuffle.
            random.shuffle(data)
            # Get batches.
            batches = [data[k:k+batch_size] \
                        for k in range(0, n, batch_size)]
            # inner = tqdm(total=len(batches), desc='Batch', position=1)
            # losses = tqdm(total=0, position=2, bar_format='{desc}')
            for b in tqdm(batches):
                # Unzip the batch.
                batch, labels = zip(*b)
                # Get activations and drives.
                activations, drives = self.forward(batch)
                # Get deltas.
                delta_w, delta_b, loss = self.back(activations, drives, labels)
                # for j, item in enumerate(delta_b):
                #     item.shape
                    # print(self.weights[j].shape)
                # Apply weight updates.
                new_weights = []
                for delta, w in zip(delta_w, self.weights):
                    new_weights.append(w - delta * lr)

                # Apply bias updates.
                new_biases = []
                for delta, b in zip(delta_b, self.biases):
                    new_biases.append(b - delta * lr)

                self.weights = new_weights
                self.biases = new_biases

                # losses.set_description_str(f'Loss: {loss}')
                losses.append(loss)
                # inner.update(1)

        plt.plot(losses)
        plt.show()

            # outer.update(1)
            # print(i)

    def eval(self, x, y):
        '''
        Evaluate model accuracy
        against test data.

        Parameters
        ---------
            x : np.array
                Test input.
            y : np.array
                Test labels.

        Returns
        ---------
            acc : float
                Test accuracy.
        '''
        # One-hot encode the labels.
        y = [self.labels[value] for value in y]
        # Flatten all input arrays.
        X = x.reshape(x.shape[0], -1)
        # Organize data for inference.
        data = list(zip(X, y))
        n = len(data)
        # Get batches.
        batches = [data[k:k+self.batch_size] \
                    for k in range(0, n, self.batch_size)]
        acc = []

        preds = []
        ims = []

        for b in batches:
            try:
                # Unzip the batch.
                batch, labels = zip(*b)
                # Get activations and drives.
                activations, _ = self.forward(batch)
                # Output
                for i, item in enumerate(batch):
                    preds.append(np.max(activations[-1][i]))
                    ims.append(item)
                    pred = np.argmax(activations[-1][i])
                    if np.argmax(labels[i]) == pred:
                        acc.append(1)
                    else:
                        acc.append(0)
            except ValueError:
                pass

        # results = sorted(list(zip(preds, ims)), key=lambda x: x[0], reverse=False)
        # for i in results[:5]:
        #     plt.imshow(i[1].reshape(28, 28))
        #     plt.show()

        return sum(acc) / len(acc)
