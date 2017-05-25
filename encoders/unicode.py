import json
import numpy as np


class UnicodeEncoder(object):
    """
    The Unicode encoder encodes a Unicode character into a sparse array of bits
    such that semantically similar characters have overlap.
    """

    def __init__(self):
        self._n = 2048
        self._w = 37
        with open('encoders/data/chars.json', 'r') as fi:
            lookup = json.load(fi)
        self._lookup = {int(key): val for key, val in lookup.items()}

    def encodeIntoArray(self, inputData, outputArray):
        """
        Encodes inputData and puts the encoded value into the numpy outputArray
        which is a 1-D array of length returned by :meth:`.getWidth`.
        Note: The numpy array is reused, so it is cleared before updatign it.
        :param inputData: The data to encode (must be a string of length 1)
        :param outputArray: numpy 1-D array of the same length returned by :meth:`.getWidth`.
        """
        if not isinstance(inputData, str):
            raise TypeError("Expected a string or unicode but got input of type %s" % type(inputData))
        if len(inputData) != 1:
            raise ValueError("Expected a string of length 1")
        outputArray[:] = 0
        outputArray[self.encodeIntoBits(inputData)] = 1

    def encodeIntoBits(self, inputData):
        """
        Encodes inputData and generates a list of ON bits in a 1-D array of length returned by :meth:`.getWidth`.
        :param inputData: The data to encode (must be a string of length 1)
        :return: A list of integers (indices of ON bits in 1-D array of length returned by :meth:`.getWidth`.)
        """
        try:
            return self._lookup[ord(inputData)]
        except KeyError:
            return []

    def getWidth(self):
        """
        Return length of encoded 1-D arrays
        :return: width of encoded arrays 
        """
        return self._n
