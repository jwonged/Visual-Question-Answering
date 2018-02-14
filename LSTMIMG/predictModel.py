'''
Created on 13 Feb 2018

@author: jwong
'''
from LSTMIMGmodel import LSTMIMGmodel
from LSTMIMG_LapConfig import LSTMIMG_LapConfig
from LSTMIMG_GPUConfig import LSTMIMG_GPUConfig
from InputProcessor import InputProcessor, TestProcessor
import pickle
import csv


def loadAndTest():
    config = LSTMIMG_LapConfig(load=True)
    #config = LSTMIMG_GPUConfig(load=True)
    
    testReader = TestProcessor(qnFile=config.testOfficialDevQns, 
                               imgFile=config.testOfficialImgFeatures, 
                               config=config)
    
    model = LSTMIMGmodel(config)
    model.loadTrainedModel()
    model.runTest(testReader)
    model.destruct()

def runValTest():
    print('Running Val Test')
    config = LSTMIMG_LapConfig(load=True)
    valTestReader = InputProcessor(config.testAnnotFile, 
                                 config.rawQnValTestFile, 
                                 config.testImgFile, 
                                 config,
                                 is_training=False)
    
    model = LSTMIMGmodel(config)
    model.loadTrainedModel()
    model.runPredict(valTestReader)
    model.destruct()

if __name__ == '__main__':
    runValTest()