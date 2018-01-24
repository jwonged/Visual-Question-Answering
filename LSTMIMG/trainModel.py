'''
Created on 15 Jan 2018

@author: jwong
'''
from LSTMIMGmodel import LSTMIMGmodel
from Config import Config
from InputProcessor import InputProcessor
import pickle

def runtrain():
    config = Config()
    trainReader = InputProcessor(config.trainAnnotFile, 
                                 config.rawQnTrain, 
                                 config.trainImgFile, 
                                 config.ansClass1000File, 
                                 config,
                                 is_training=True)
    
    valReader = InputProcessor(config.valAnnotFile, 
                                 config.rawQnValTestFile, 
                                 config.valImgFile, 
                                 config.ansClass1000File, 
                                 config,
                                 is_training=True)
    
    #dumReader = DummyReader()
    
    model = LSTMIMGmodel(config)
    model.construct()
    model.train(trainReader, valReader)
    model.destruct()
            
class DummyReader():
    def __init__(self):
        file = '/media/jwong/Transcend/VQADataset/DummySets/dummyTupleBatchesLSTMIMG.pkl'
        with open(file, 'rb') as jFile:
            self.tupList = pickle.load(jFile)
    
    def getNextBatch(self, batch_size):
        for tup in self.tupList:
            yield tup
    
    def getWholeBatch(self):
        return self.tupList[80]
        
if __name__ == '__main__':
    runtrain()