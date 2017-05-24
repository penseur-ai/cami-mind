import numpy as np


class Classifier:

    def __init__(self, columnDim, alpha):

        if alpha < 0 or alpha > 1:
            raise ValueError("alpha must be a number between 0 and 1")

        self._columnDim = columnDim
        self._alpha = alpha
        self._maxTimesSeen = int(1.0 / alpha)
        self._history = []
        self._lookup = {}
        self._revlookup = {}

    def record(self, inputData, selectedColumns):
        columns = np.zeros(self._columnDim, dtype=np.int64)
        columns[selectedColumns] = 1
        try:
            index = self._lookup[inputData]
            self._history[index] += columns
            if np.count_nonzero(self._history[index] > self._maxTimesSeen):
                temp = self._history[index]
                minTimesSeen = np.min(temp[temp > 0])
                if minTimesSeen != self._maxTimesSeen:  # Need TODO something different here
                    temp[temp > 0] -= minTimesSeen
                    self._history[index] = temp

        except KeyError:
            self._history.append(columns)
            self._lookup[inputData] = len(self._history) - 1
            self._revlookup[len(self._history) - 1] = inputData

    def infer(self, selectedColumns):
        columns = np.zeros(self._columnDim, dtype=np.int8)
        columns[selectedColumns] = 1
        history = np.array(self._history)
        overlapScores = np.count_nonzero(np.multiply(history, columns), axis=1)
        # find probability distribution
        # TODO: implement prob.dist
        # return tuple of lists of historical input data and probability distribution for inferences
        inputs = [self._revlookup[i] for i in range(len(overlapScores))]
        return inputs

    def get_alpha(self):
        """Returns value of alpha parameter"""
        return self._alpha

    def set_alpha(self, alpha):
        """Sets the value of alpha parameter"""
        if alpha < 0 or alpha > 1:
            raise ValueError("alpha must be a number between 0 and 1")
        self._alpha = alpha
        self._maxTimesSeen = int(1.0 / alpha)
