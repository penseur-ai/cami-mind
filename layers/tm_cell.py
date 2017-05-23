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
        # self._segmentPotentials = []
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

        matchingSynapses = (np.array(self._segmentPermanences) > 0) * activeStates
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
                self._segmentPermanences[segment] = np.clip(self._segmentPermanences[segment] + updates, 0.0, 1.0)

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
                self._segmentPermanences[segment] = np.clip(self._segmentPermanences[segment] + updates, 0.0, 1.0)

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
                self._segmentPermanences[segment] = np.clip(self._segmentPermanences[segment] - updates, 0.0, 1.0)

    def createSegment(self, maxNewSynapses, prevWinnerCells, initialPermanence):
        """
        Creates new segment and synapses on segment to previous winner cells
        :param maxNewSynapses: (int) maximum number of new synapses to create
        :param prevWinnerCells: (array-like) list of previous winner cells to adapt to
        :param initialPermanence: (float) permanence value for new synapsess
        """
        newSynapsesCount = min(maxNewSynapses, len(prevWinnerCells))
        if newSynapsesCount > 0:
            synapses = np.zeros(self._regionDim, dtype=np.float32).flatten()
            eligibleSynapses = np.array(prevWinnerCells)
            np.random.shuffle(eligibleSynapses)
            for i in range(newSynapsesCount):
                synapses[eligibleSynapses[i]] = initialPermanence
            self._segmentPermanences.append(synapses)

    def addSynapses(self, numberOfSynapses, prevWinnerCells, initialPermanence):
        return NotImplementedError()

    def adaptSegment(self, prevWinnerCells, initialPermanence):
        return NotImplementedError()

    def adaptActiveSegments(self, prevActiveCells, maxNewSynapses, prevWinnerCells, initialPermanence,
                            permanenceInc, permanenceDec):

        if len(self._activeSegments) > 0:
            # self.updateActiveSegments(prevActiveCells, permanenceInc, permanenceDec)
            updates = np.full(self._regionDim, -permanenceDec).flatten()
            updates[prevActiveCells] = permanenceInc
            updates = updates.reshape(self._regionDim)
            for segment in self._matchingSegments:
                self._segmentPermanences[segment] = np.clip(self._segmentPermanences[segment] + updates, 0.0, 1.0)

            activeSynapsesCount = np.count_nonzero(updates)  # TODO check logic
            newSynapsesCount = maxNewSynapses - activeSynapsesCount
            if newSynapsesCount > 0:
                eligibleSynapses = np.logical_and(updates > 0, self._segmentPermanences == 0)
                eligibleSynapses = np.transpose(np.nonzero(eligibleSynapses))
                np.random.shuffle(eligibleSynapses)
                for i in range(newSynapsesCount):
                    self._segmentPermanences[segment][eligibleSynapses[i]] = initialPermanence

        return NotImplementedError()

    def adaptMatchingSegemnt(self, segment, prevWinnerCells, maxNewSynapses, initialPermanence,
                             permanenceInc, permanenceDec):
        """
        Updates permanence values for segment and creates new synapses on segment to previous winner cells
        :param segment: (integer) matching segment to adapt
        :param prevWinnerCells: (array-like) list of previous winner cells to adapt to
        :param maxNewSynapses: (int) maximum number of new synapses to create
        :param initialPermanence: (float) permanence value for new synapses
        :param permanenceInc: (float) value to increment active synapses
        :param permanenceDec: (float) value to decrement inactive synapses
        """
        updates = np.full(self._regionDim, permanenceDec, dtype=np.int8).flatten()
        updates[prevWinnerCells] = permanenceInc
        updates = updates.reshape(self._regionDim) * (self._segmentPermanences[segment] > 0)

        self._segmentPermanences[segment] = np.clip(self._segmentPermanences[segment] + updates, 0.0, 1.0)

        activeSynapsesCount = np.count_nonzero(updates)  # TODO check logic
        newSynapsesCount = maxNewSynapses - activeSynapsesCount
        if newSynapsesCount > 0:
            eligibleSynapses = np.logical_and(updates > 0, self._segmentPermanences == 0)
            eligibleSynapses = np.transpose(np.nonzero(eligibleSynapses))
            np.random.shuffle(eligibleSynapses)
            for i in range(newSynapsesCount):
                self._segmentPermanences[segment][eligibleSynapses[i]] = initialPermanence

    def getActivePotentials(self, activeCells):
        """
        Returns the counts of active matching synapses for each segment 
        :param activeCells: array_like list of active cells indices in region
        :return: numpy array
        """

        if len(self._segmentPermanences) < 1:  # no segments on this cell
            return [0]

        activeStates = np.zeros(self._regionDim, dtype=np.int8).flatten()
        activeStates[activeCells] = 1
        activeStates = activeStates.reshape(self._regionDim)

        # activePotentials = np.sum((np.array(self._segmentPermanences) > 0) * activeStates, axis=(1, 2))
        activePotentials = np.count_nonzero((self._segmentPermanences * activeStates), axis=(1, 2))
        return activePotentials

    def getNumberOfSegments(self):
        """Returns the number of segments on this cell"""
        return len(self._segmentPermanences)
