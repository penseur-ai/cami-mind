import numpy as np


class FeedForwardClassifier:

    def __init__(self, columnDim):
        self._name = "FeedForwardClassifier"
        self._weights = [np.zeros(columnDim, dtype=np.float32)]
        self._labels = [None]
