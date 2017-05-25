import numpy as np


class FeedForwardClassifier:
    """
    This class implements a single layer feed forward neural network classifier for SDRs.
    The classifier grows with new unseen data.
    """

    def __init__(self, columnDim, alpha):
        """
        Constructs a feed-forward neural network classifier
        :param columnDim: (int) number columns in input
        :param alpha: (float) learning rate
        """
        self._columnDim = columnDim
        self._weights = np.random.normal(size=columnDim)
        self._alpha = alpha
        self._lookup = {None: 0}
        self._revlookup = {0: None}
        self._history = None
        self._bits = None

    def record(self, inputData, columns):
        """
        Records and learns to predict inputData based on columns at t-l 
        :param inputData: data value to be predicted
        :param columns: (array-like) columns with ON bits
        """
        if self._history is not None:

            try:
                index = self._lookup[inputData]
            except KeyError:
                if len(self._weights.shape) == 1:
                    index = 1
                else:
                    index = self._weights.shape[0]
                self._weights = np.vstack((self._weights,
                                           np.random.normal(scale=np.sqrt(1.0/(index + 1)),
                                                            size=self._columnDim)))
                self._revlookup[len(self._lookup)] = inputData
                self._lookup[inputData] = len(self._lookup)

            _, probabilities = self._infer(self._history, learn=True)

            self.learn(probabilities, index)
            self._bits = None

        self._history = columns

    def _infer(self, columns, learn=True):
        """
        Given a set of columns in the input space, infer the represented data
        :param columns: (array-like) columns with ON bits 
        :param learn: if True, save bits and z scores for learning
        :return: tuple of labels and their probabilities
        """
        self._bits = np.zeros(self._columnDim)
        self._bits[columns] = 1
        z = np.exp(np.dot(self._weights, self._bits))
        probabilities = z / np.sum(z)
        labels = [self._revlookup[i] for i in range(len(self._weights))]
        if not learn:
            self._bits = None
        return labels, probabilities

    def infer(self, columns):
        """
        Given a set of columns in the input space, infer the represented data
        :param columns: (array-like) columns with ON bits 
        :return: tuple of labels and their probabilities
        """
        return self._infer(columns, learn=False)

    def learn(self, probabilities, label):
        """
        Update weights via back-propagation
        :param probabilities: probabilities of output labels
        :param label: actual output label
        """
        loss = -np.log(probabilities[label])
        gradients = probabilities
        gradients[label] -= 1
        deltas = np.matmul(gradients.reshape((-1, 1)), self._bits.reshape((1, -1)))
        self._weights += -self._alpha * deltas
