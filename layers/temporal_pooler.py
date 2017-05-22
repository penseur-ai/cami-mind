import numpy as np
from mind.layers.tm_cell import TMCell


class TemporalPooler:
    """
    This class implements the Temporal Memory algorithm learning sequences of sparse
    distributed representations of sensory inputs. A layer of temporal memory performs
    prediction and inference of sequences using active columns from the immediately
    preceding layer.
    """

    def __init__(self, cell, columnDim, cellsPerColumn, maxSegmentsPerCell, maxSynapsesPerSegment, minActive,
                 activationThreshold, minPermanence, permanenceInc, permanenceDec, seed=45):

        np.random.seed(seed)

        self._columnDim = columnDim
        self._cellsPerColumn = cellsPerColumn
        self._n = columnDim * cellsPerColumn
        self._maxSegmentsPerCell = maxSegmentsPerCell
        self._maxSynapsesPerSegment = maxSynapsesPerSegment
        self._minActive = minActive
        self._activationThreshold = activationThreshold
        self._minPermanence = minPermanence
        self._permanenceInc = permanenceInc
        self._permanenceDec = permanenceDec

        self._cells = [[cell((columnDim, cellsPerColumn), minPermanence, activationThreshold,
                            minActive, maxSegmentsPerCell, maxSynapsesPerSegment)
                        for j in range(cellsPerColumn)]
                       for i in range(columnDim)]

        self._activeCells = []
        self._winnerCells = []

    def compute(self, activeColumns, learn=True):
        """
        
        :param activeColumns: 
        :param learn: 
        :return: 
        
        Algorithm:
        ACTIVE COLUMNS W PREDICTIVE CELLS
        Take each active column's active segments from t-l [cols,cells,segments] (Cells in predicted state) [cols,cells] 
            Predictive cells are active AND winner cells [cols,cells]
            Learn from previous active cells (t-l) [cols,cells] w/ [cols,cells,segments,cols,cells]
            Grow to previous winner cells (t-l) [cols,cells] w/ [cols,cells,segments,cols,cells]
        
        ACTIVE BURSTING COLUMNS
        Take each active column's matching segments from t-l [cols,cells,segments]
            All cells are active
            Find winner cell in each bursting column
                Cell with most active matching segment from t-l OR
                Cell with least number of segments
            Learn from previous active cells (t-l) [cols,cells] w/ [cols,cells,segments,cols,cells]
            Grow to previous winner cells (t-l) [cols,cells] w/ [cols,cells,segments,cols,cells]
            
        INACTIVE COLUMNS W PREDICTIVE CELLS
        Take each inactive column's matching segments from t-l [cols,cells,segments]
            Learn from previous active cells (t-l) [cols,cells] w/ [cols,cells,segments,cols,cells]
            
        Find active segments using current active cells (t) [cols,cells,segments,cols,cells] w/ [cols,cells] 
        Find matching segments using current active cells (t) [cols,cells,segments,cols,cells] w/ [cols,cells]
        """

        prevActiveCells = self._activeCells
        prevWinnerCells = self._winnerCells

        predictiveStates = np.array([[cell.predictive() for cell in col] for col in self._cells], dtype=np.int8)
        columns = np.zeros((self._columnDim, self._cellsPerColumn), dtype=np.int8)
        columns[activeColumns, :] = 1
        winnerCells = activeCells = np.logical_and(columns, predictiveStates)
        temp = np.sum(self._activeCells, axis=1) > 0
        burstingColumns = np.where(temp != columns[0])[0]
        activeCells[burstingColumns, :] = 1
        self._activeCells = np.nonzero(activeCells.flatten())[0]
        self._findWinnerCells(winnerCells, burstingColumns)

        if learn:
            pass  # with winner cells?

        for col in self._cells:
            for cell in col:
                cell.activateSegments(self._activeCells)

        predictiveStates = np.array([[cell.predictive() for cell in col] for col in self._cells], dtype=np.int8)
        predictiveCells = np.nonzero(predictiveStates.flatten())[0]

        return self._activeCells, predictiveCells


        predictiveStates = (np.sum(self._activeSegments, axis=2) > 0).astype(np.int8)
        columns = np.zeros((self._columnDim, self._cellsPerColumn), dtype=np.int8)
        columns[activeColumns, :] = 1
        self._winnerCells = self._activeCells = np.logical_and(columns, predictiveStates)
        temp = np.sum(self._activeCells, axis=1) > 0
        burstingColumns = np.where(temp != columns[0])[0]
        self._activeCells[burstingColumns, :] = 1

        self._findWinnerCells(burstingColumns)  # uses previous matching segments

        connectedSynapses = (self._permanences >= self._minPermanence) * self._activeCells
        self._activeSegments = np.sum(connectedSynapses, axis=(3, 4)) >= self._activationThreshold
        # self._predictedCells = (np.sum(self._activeSegments, axis=2) > 0).astype(np.int8)

        matchingSynapses = self._potentials * self._activeCells  # self._potentials should equal (self._permanences > 0)
        self._matchingSegments = np.sum(matchingSynapses, axis=(3, 4)) >= self._minActive

        if learn:
            activeUpdates = columns * (self._permanenceInc + self._permanenceDec) - self._permanenceDec
            inactiveUdpates = columns * self._permanenceDec

        """
        Return array of indices instead of ndarrays? Easier to persist
        But which is worse? the extra computation time or extra storage
        If you save indices each compute cycle requires converting indices into ndarrays and back again as well as
        reshape operations
        """

        return self._activeCells, predictiveCells

    def _findWinnerCells(self, winnerCells, burstingColumns):
        """
        Find winner cell in each bursting column
                Cell with most active matching segment from t-l OR
                Cell with least number of segments
        :param winnerCells: 
        :param burstingColumns: 
        :return: 
        """
        self.winnerCells = np.nonzero(winnerCells.flatten())[0]
        return NotImplementedError("findWinnerCells not implemented")

    def getWinnerCells(self, asarray=False):
        """
        Returns the winner cells in the region
        :param asarray: If True, returns array of winner cell states; otherwise, returns list of indices
        :return: numpy array
        """
        if asarray:
            winnerCells = np.zeros(self._n, dtype=np.int8)
            winnerCells[self._winnerCells] = 1
            return winnerCells
        else:
            return self._winnerCells

    def getActiveCells(self, asarray=False):
        """
        Returns the active cells in the region
        :param asarray: If True, returns array of active cell states; otherwise, returns list of indices
        :return: numpy array
        """
        if asarray:
            activeCells = np.zeros(self._n, dtype=np.int8)
            activeCells[self._activeCells] = 1
            return activeCells
        else:
            return self._activeCells

    def getPredictedCells(self, asarray=True):
        """
        Returns the cells in the region in the predictive state
        :param asarray: If True, returns array of predictive cell states; otherwise, returns list of predictive cells
        :return: numpy array
        """
        predictiveStates = np.array([[cell.predictive() for cell in col] for col in self._cells], dtype=np.int8)
        if asarray:
            return predictiveStates.flatten()
        else:
            return np.nonzero(predictiveStates.flatten())[0]

    def getPredictedColumns(self, asarray=True):
        """
        Returns the predicted columns in a region
        :param asarray: If True, returns array of predictive column states; otherwise, returns list of predicted columns
        :return: numpy array
        """
        predictiveStates = np.array([[cell.predictive() for cell in col] for col in self._cells], dtype=np.int8)
        prediction = np.sum(predictiveStates, axis=1) > 0
        if asarray:
            return prediction
        else:
            return np.nonzero(prediction)[0]

    def getWidth(self):
        """Returns the total number of cells in the region"""
        return self._n

    def getPermanenceIncrement(self):
        """Returns the learning increment value for synapse permanences"""
        return self._permanenceInc

    def getPermanenceDecrement(self):
        """Returns the learning decrement value for synapse permanences"""
        return self._permanenceDec

    def getPermanenceThreshold(self):
        """Returns the permanence threshold for connected synapses"""
        return self._minPermanence

    def setPermanenceIncrement(self, permanenceInc):
        """Sets the learning increment value for synapse permanences"""
        self._permanenceInc = permanenceInc

    def setPermanenceDecrement(self, permanenceDec):
        """Sets the learning decrement value for  synapse permanences"""
        self._permanenceDec = permanenceDec

    def setPermanenceThreshold(self, minPermanence):
        """Returns the permanence threshold for connected synapses"""
        self._minPermanence = minPermanence
