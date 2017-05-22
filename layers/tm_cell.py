import numpy as np


class TMCell:

    def __init__(self, regionDim, minPermanence, activationThreshold,
                 minActive, maxSegmentsPerCell, maxSynapsesPerSegment):
        self._regionDim = regionDim
        self._minPermanence = minPermanence
        self._activationThreshold = activationThreshold
        self._minActive = minActive
        self._maxSegmentsPerCell = maxSegmentsPerCell
        self._maxSynapsesPerSegment = maxSynapsesPerSegment
        self._segmentPotentials = []
        self._segmentPermanences = []
        self._activeSegments = []
        self._matchingSegments = []

    def matching(self):
        """Returns matching state of cell"""
        return len(self._matchingSegments) > 0

    def predictive(self):
        """Returns predictive state of cell"""
        return len(self._activeSegments) > 0

    def activateSegments(self, activeCells):
        """
        Using current active cells find cell's segment activity
        :param activeCells: List of indices of active cells in region
        """
        activeStates = np.zeros(self._regionDim, dtype=np.int8).flatten()
        activeStates[activeCells] = 1
        activeStates = activeStates.reshape(self._regionDim)

        connectedSynapses = (np.array(self._segmentPermanences) >= self._minPermanence) * activeStates
        self._activeSegments = np.nonzero(np.sum(connectedSynapses, axis=(1, 2)) >= self._activationThreshold)[0]

        matchingSynapses = np.array(self._segmentPotentials) * activeStates
        self._matchingSegments = np.nonzero(np.sum(matchingSynapses, axis=(1, 2)) >= self._minActive)[0]

    def updateMatchingSegments(self, prevWinnerCells, permanenceInc, permanenceDec):
        """
        
        :param prevWinnerCells: 
        :param permanenceInc: 
        :param permanenceDec: 
        :return: 
        """
        if len(self._matchingSegments) > 0:
            updates = np.full(self._regionDim, -permanenceDec).flatten()
            updates[prevWinnerCells] = permanenceInc
            updates = updates.reshape(self._regionDim)
            for segment in self._matchingSegments:
                self._segmentPermanences[segment] += updates

    def updateActiveSegments(self, prevActiveCells, permanenceInc, permanenceDec):
        """
        
        :param prevActiveCells: 
        :param permanenceInc: 
        :param permanenceDec: 
        :return: 
        """
        if len(self._activeSegments) > 0:
            updates = np.full(self._regionDim, -permanenceDec).flatten()
            updates[prevActiveCells] = permanenceInc
            updates = updates.reshape(self._regionDim)
            for segment in self._matchingSegments:
                self._segmentPermanences[segment] += updates

    def punishMatchingSegments(self, prevActiveCells, permanenceDec):
        """
        
        :param prevActiveCells: 
        :param permanenceDec: 
        :return: 
        """
        if len(self._matchingSegments) > 0:
            updates = np.zeros(self._regionDim).flatten()
            updates[prevActiveCells] = permanenceDec
            updates = updates.reshape(self._regionDim)
            for segment in self._matchingSegments:
                self._segmentPermanences[segment] -= updates

    def createSegment(self, numberOfSynapses, prevWinnerCells, initialPermanence):
        pass

    def addSynapses(self, numberOfSynapses, prevWinnerCells, initialPermanence):
        pass

    def adaptSegment(self, prevWinnerCells, initialPermanence):
        pass
