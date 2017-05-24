import numpy as np


class TMCell:
    """
    This is the base class for Temporal Memory cell.
    """

    def activateSegments(self, activeCells):
        """
        Using current active cells find cell's segment activity
        :param activeCells: List of indices of active cells in region
        """
        raise NotImplementedError()

    def adaptSegment(self, segment, previousCells, maxNewSynapses, initialPermanence,
                     permanenceInc, permanenceDec):
        """
        Updates permanence values for segment and creates new synapses on segment to previous cells
        :param segment: (integer) matching segment to adapt
        :param previousCells: (array-like) list of previous cells to adapt to
        :param maxNewSynapses: (int) maximum number of new synapses to create
        :param initialPermanence: (float) permanence value for new synapses
        :param permanenceInc: (float) value to increment active synapses
        :param permanenceDec: (float) value to decrement inactive synapses
        """

        raise NotImplementedError()

    def adaptActiveSegments(self, prevActiveCells, maxNewSynapses, prevWinnerCells, initialPermanence,
                            permanenceInc, permanenceDec):
        """
        Updates permanence values for active segments and creates new synapses to previous winner cells
        :param prevActiveCells: (array-like) list of previous active cells to adapt to
        :param maxNewSynapses: (int) maximum number of new synapses to create
        :param prevWinnerCells: (array-like) list of previous winner cells to grow to
        :param initialPermanence: (float) permanence value for new synapses
        :param permanenceInc: (float) value to increment active synapses
        :param permanenceDec: (float) value to decrement inactive synapses 
        """

        raise NotImplementedError()

    def addSynapses(self, segment, previousCells, maxNewSynapses, initialPermanence):
        """
        Creates new synapses on segment to previous cells
        :param segment: (integer) matching segment to adapt
        :param previousCells: (array-like) list of previous cells to adapt to
        :param maxNewSynapses: (int) maximum number of new synapses to create
        :param initialPermanence: (float) permanence value for new synapses
        """
        raise NotImplementedError()

    def createSegment(self, maxNewSynapses, prevWinnerCells, initialPermanence):
        """
        Creates new segment and synapses on segment to previous winner cells
        :param maxNewSynapses: (int) maximum number of new synapses to create
        :param prevWinnerCells: (array-like) list of previous winner cells to adapt to
        :param initialPermanence: (float) permanence value for new synapsess
        """
        raise NotImplementedError()

    def punishMatchingSegments(self, prevActiveCells, permanenceDec):
        """
        Updates permanence values for matching segments for inactive columns
        :param prevActiveCells: (array-like) list of previous active cells to update
        :param permanenceDec: (float) permanence value to decrement previous active cells
        """
        raise NotImplementedError()

    def getActivePotentials(self, activeCells):
        """
        Returns the counts of active matching synapses for each segment 
        :param activeCells: array_like list of active cells indices in region
        :return: numpy array
        """
        raise NotImplementedError()

    def getNumberOfSegments(self):
        """Returns the number of segments on this cell"""
        raise NotImplementedError()

    def matching(self):
        """Returns matching state of cell"""
        raise NotImplementedError()

    def predictive(self):
        """Returns predictive state of cell"""
        raise NotImplementedError()
