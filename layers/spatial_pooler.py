import numpy as np

class SpatialPooler():
    """
    This class implements the Spatial Pooling algorithm forming sparse distributed representations of sensory inputs
    """

    def __init__(self, inputDim, columnDim, numActiveCols, pot_pct, minPermanence, activeInc, inactiveDec, seed=23):
        """
        Constructs a Spatial Pooling layer and initializes variables
        :param inputDim: The dimensions of encoded input vectors
        :param columnDim: The number of columns in the Spatial Pooler; also length of output
        :param numActiveCols: The number of active columns resulting from spatial pooling
        :param pot_pct: The percentage of potential synapses for each column
        :param minPermanence: The permanence threshold for potentially connected synapses
        :param activeInc: The increment value for active synapse permanence learning
        :param inactiveDec: The decrement value for inactive synapse permanence learning
        :param seed: The seed for the random number generator
        """

        np.random.seed(seed)

        self._inputDim = inputDim
        self._n = columnDim
        self._w = numActiveCols
        self._pot_pct = pot_pct
        self._minPermanence = minPermanence
        self._activeInc = activeInc
        self._inactiveDec = inactiveDec

        self._permanences = np.zeros((columnDim, inputDim), dtype=np.float32)
        self._potentials = np.zeros((columnDim, inputDim), dtype=np.int8)

        self._numPotentials = int(pot_pct * inputDim + 0.5)

        for i in range(columnDim):
            bits = np.random.randint(0, inputDim, (1, self._numPotentials), dtype=np.int64)
            self._potentials[i, bits] = 1
            self._permanences[i, bits] = np.random.normal(loc=minPermanence,
                                                          scale=0.25*minPermanence,
                                                          size=self._numPotentials)


    def compute(self, encodedInput, learn=True, asarray=False):
        """
        Returns a sparse distributed representation of the input as indices of active columns or if 'asarray'
        is True, as a 1-D array of length returned by :meth:`.getWidth`. If 'learn' is set to True, updates
        permanences of active columns
        :param encodedInput: A binary numpy array 
        :param learn: (default=True) Indicates whether learning should be performed and permanence values updated
        :param asarray: (default=False) if True, returns a 1-D array of length returned by :meth:`.getWidth`.
        :return: List of integers OR 1-D array of length returned by :meth:`.getWidth`.
        """

        if not isinstance(encodedInput, np.ndarray):
            raise TypeError("Input must be a numpy array but got input of type %s" % type(encodedInput))
        if encodedInput.size != self._inputDim:
            raise ValueError("Input dimensions do not match. Expecting %d but got %d" % (self._inputDim,
                                                                                         encodedInput.size))


        overlapScores = np.sum(np.multiply(self._permanences >= self._minPermanence, encodedInput), axis=1)
        activeCols = np.argsort(overlapScores)[:self._w]

        if learn:
            updates = encodedInput * (self._activeInc + self._inactiveDec) - self._inactiveDec
            updates = np.multiply(self._potentials[activeCols], updates)
            self._permanences[activeCols] = np.clip(self._permanences[activeCols] + updates, 0.0, 1.0)

        if asarray:
            columns = np.zeros(self._n, dtype=np.int8)
            columns[activeCols] = 1
            return columns
        else:
            return activeCols


    def getWidth(self):
        """Returns the number of columns in the Spatial Pooler"""
        return self._n

    def getActiveIncrement(self):
        """Returns the learning increment value for active synapse permanences"""
        return self._activeInc

    def getInactiveDecrement(self):
        """Returns the learning decrement value for inactive synapse permanences"""
        return self._inactiveDec

    def getPotentialPercentage(self):
        """Returns the number of potential synapses for each column as a percentage"""
        return self._pot_pct

    def getPermanenceThreshold(self):
        """Returns the permanence threshold for potentially connected synapses"""
        return self._minPermanence

    def setActiveIncrement(self, activeInc):
        """Sets the learning increment value for active synapse permanences"""
        self._activeInc = activeInc

    def setInactiveDecrement(self, inactiveDec):
        """Sets the learning decrement value for inactive synapse permanences"""
        self._inactiveDec = inactiveDec

    def setPermanenceThreshold(self, minPermanence):
        """Returns the permanence threshold for potentially connected synapses"""
        self._minPermanence = minPermanence
