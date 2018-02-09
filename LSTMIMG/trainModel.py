'''
Created on 15 Jan 2018

@author: jwong
'''
from LSTMIMGmodel import LSTMIMGmodel
from LSTMIMG import LSTMIMG_LapConfig
from LSTMIMG_GPUConfig import LSTMIMG_GPUConfig
from InputProcessor import InputProcessor
import pickle
import csv

def runtrain():
    #config = LSTMIMG_LapConfig()
    config = LSTMIMG_GPUConfig()
    
    trainReader = InputProcessor(config.trainAnnotFile, 
                                 config.rawQnTrain, 
                                 config.trainImgFile, 
                                 config,
                                 is_training=True)
    
    valReader = InputProcessor(config.valAnnotFile, 
                                 config.rawQnValTestFile, 
                                 config.valImgFile, 
                                 config,
                                 is_training=False)
    
    #dumReader = DummyReader(config)
    
    model = LSTMIMGmodel(config)
    model.construct()
    model.train(trainReader, valReader)
    #model.train(dumReader, dumReader)
    model.destruct()

def makeSmallDummyData():
    config = LSTMIMG_LapConfig()
    trainReader = InputProcessor(config.trainAnnotFile, 
                                 config.rawQnTrain, 
                                 config.trainImgFile, 
                                 config.ansClass1000File, 
                                 config,
                                 is_training=True)
    
    #dumReader = DummyReader()
    dummyData = []
    for i, (batch) in enumerate(
            trainReader.getNextBatch(32)):
        if i==100:
            break
        dummyData.append(batch)
    
    print('Completed producing dataset of size {}'.format(len(dummyData)))
    file = '/media/jwong/Transcend/VQADataset/DummySets/dummyTupleBatchesLSTMIMG.pkl'
    with open(file, 'wb') as f:
            pickle.dump(dummyData, f, protocol=pickle.HIGHEST_PROTOCOL)
    print('Printed to file')
    


class DummyReader():
    def __init__(self, config):
        file = '/media/jwong/Transcend/VQADataset/DummySets/dummyTupleBatchesLSTMIMG.pkl'
        with open(file, 'rb') as jFile:
            print('Reading {}'.format(file))
            self.tupList = pickle.load(jFile)
        print('Reading ' + config.ansClass1000File)
        self.mapAnsToClass, self.classToAnsMap = self._loadAnsMap(config.ansClass1000File)
    
    def getNextBatch(self, batch_size):
        for tup in self.tupList:
            yield tup
    
    def _loadAnsMap(self, ansClassFile):
        #loads mapping: ans --> ans class index
        with open(ansClassFile, 'rb') as ansFile:
            reader = csv.reader(ansFile, delimiter=',')
            ansVec = next(reader)
        classToAnsMap = {}
        ansClassMap = {}
        for classIndex, word in enumerate(ansVec):
            word = word.strip()
            ansClassMap[word] = classIndex
            classToAnsMap[classIndex] = word
        print('Read in answer mapping with {} answers'.format(len(ansClassMap)))
        return ansClassMap, classToAnsMap
    
    def getAnsMap(self):
        return self.classToAnsMap
        
if __name__ == '__main__':
    runtrain()