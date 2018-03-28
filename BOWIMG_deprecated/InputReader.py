'''
Created on 18 Dec 2017

@author: jwong
'''
import pickle

class InputReader(object):
    def __init__(self, xTrainFile, yTrainFile):
        print('Reading ' + xTrainFile)
        with open(xTrainFile, 'rb') as jFile:
            self.trainX = pickle.load(jFile)
        print('Reading ' + yTrainFile)
        with open(yTrainFile, 'rb') as jFile:
            self.trainY = pickle.load(jFile)
        self.size = len(self.trainX)
        self.index = 0
        self.epoch = 0

    def getEpochSize(self):
        return self.size

    def getIndexInEpoch(self):
        return self.index

    def getEpoch(self):
        return self.epoch
    
    def getWholeBatch(self):
        return self.trainX, self.trainY

    def getNextXYBatch(self, batchSize):
        if self.index > self.size: 
            #start on new epoch
            self.index = 0
            self.epoch += 1
            
        start = self.index
        self.index += batchSize
        end = self.index
        return self.trainX[start:end], self.trainY[start:end]