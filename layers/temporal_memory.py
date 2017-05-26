import numpy as np
from layers.tm_cell import TMCell


class TemporalMemory:
    """
    This class implements the Temporal Memory algorithm learning sequences of sparse
    distributed representations of sensory inputs. A layer of temporal memory performs
    prediction and inference of sequences using active columns from the immediately
    preceding layer.
    """

    def __init__(self, tm_cell, columnDim, cellsPerColumn, maxSegmentsPerCell, maxSynapsesPerSegment,
                 minActive, activationThreshold, minPermanence, initialPermanence, maxNewSynapses,
                 permanenceInc, permanenceDec, seed=45):
        """
        Constructs a Temporal Memory layer 
        :param tm_cell: 
        :param columnDim: 
        :param cellsPerColumn: 
        :param maxSegmentsPerCell: 
        :param maxSynapsesPerSegment: 
        :param minActive: 
        :param activationThreshold: 
        :param minPermanence: 
        :param initialPermanence: 
        :param maxNewSynapses: 
        :param permanenceInc: 
        :param permanenceDec: 
        :param seed: 
        """

        if not isinstance(tm_cell, type(TMCell)):
            raise TypeError("Expected a TMCell class type not type %s" % type(tm_cell))

        if columnDim <= 0 and not len(columnDim):
            raise ValueError("Number of columns or column dimensions must be greater than 0")

        if cellsPerColumn <= 0:
            raise ValueError("Number of cells per column must be greater than 0")

        np.random.seed(seed)

        self._columnDim = columnDim
        self._cellsPerColumn = cellsPerColumn
        self._n = columnDim * cellsPerColumn
        self._maxSegmentsPerCell = maxSegmentsPerCell
        self._maxSynapsesPerSegment = maxSynapsesPerSegment
        self._minActive = minActive
        self._activationThreshold = activationThreshold
        self._minPermanence = minPermanence
        self._initialPermanence = initialPermanence
        self._maxNewSynapses = maxNewSynapses
        self._permanenceInc = permanenceInc
        self._permanenceDec = permanenceDec

        self._cells = [[tm_cell((columnDim, cellsPerColumn), minPermanence, activationThreshold,
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
        self._winnerCells = (np.flatnonzero(winnerCells)).tolist()
        predictedCells = np.transpose(np.nonzero(activeCells))
        activeCellsPerColumn = np.sum(activeCells, axis=1) > 0
        burstingColumns = np.where(activeCellsPerColumn != columns[0])[0]
        activeCells[burstingColumns, :] = 1
        self._activeCells = (np.flatnonzero(activeCells)).tolist()
        self._findWinnerCells(burstingColumns, prevWinnerCells, learn)

        if learn:
            for i, j in predictedCells:  # for active predicted cells
                self._cells[i][j].adaptActiveSegments(prevActiveCells,
                                                      self._maxNewSynapses,
                                                      prevWinnerCells,
                                                      self._initialPermanence,
                                                      self._permanenceInc,
                                                      self._permanenceDec)

            inactivePredictedCells = np.logical_and(np.logical_not(columns), predictiveStates)
            inactivePredictedCells = np.transpose(np.nonzero(inactivePredictedCells))
            for i, j in inactivePredictedCells:
                self._cells[i][j].punishMatchingSegments(prevActiveCells, self._permanenceDec)

        for col in self._cells:
            for cell in col:
                cell.activateSegments(self._activeCells)

        predictiveStates = np.array([[cell.predictive() for cell in col] for col in self._cells], dtype=np.int8)
        predictiveCells = (np.flatnonzero(predictiveStates)).tolist()

        return self._activeCells, predictiveCells

    def _findWinnerCells(self, burstingColumns, prevWinnerCells, learn):
        """
        Find winner cell in each bursting column
                Cell with most active matching segment from t-l OR
                Cell with least number of segments
        :param burstingColumns: 
        :param prevWinnerCells: 
        :param learn: 
        :return: 
        """

        for column in burstingColumns:
            cellActivePotentials = []  # sequence of counts of active matching potentials for each cell
            maxActivePotentials = []  # max numbers of active matching potentials for each cell
            for i in range(self._cellsPerColumn):
                cellActivePotentials.append(self._cells[column][i].getActivePotentials(self._activeCells))
                maxActivePotentials.append(max(cellActivePotentials[-1]))

            if np.max(maxActivePotentials) > 0:
                # winner cell is cell with most active matching segment
                winnerCell = np.array(maxActivePotentials).argmax()
                if learn:
                    segment = np.argmax(cellActivePotentials[winnerCell])
                    self._cells[column][winnerCell].adaptSegment(segment,
                                                                 prevWinnerCells,
                                                                 self._maxNewSynapses,
                                                                 self._initialPermanence,
                                                                 self._permanenceInc,
                                                                 self._permanenceDec)

            else:
                # winner cell is cell with least number of segments
                segmentCounts = [self._cells[column][i].getNumberOfSegments() for i in range(self._cellsPerColumn)]
                winnerCell = np.array(segmentCounts).argmin()

                if learn:
                    self._cells[column][winnerCell].createSegment(self._maxNewSynapses,
                                                                  prevWinnerCells,
                                                                  self._initialPermanence)

            self._winnerCells.append(int(column * self._cellsPerColumn + winnerCell))

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
            return np.flatnonzero(predictiveStates)

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
            return np.flatnonzero(prediction)

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
