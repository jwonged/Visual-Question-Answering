'''
Created on 13 Feb 2018

@author: jwong
'''
from AttentionModel import AttentionModel
from Attention_LapConfig import Attention_LapConfig
from Attention_GPUConfig import Attention_GPUConfig
from InputProcessor import InputProcessor, TestProcessor
import pickle
import csv


def loadOfficialTest():
    config = Attention_LapConfig(load=True)
    
    testReader = TestProcessor(qnFile=config.testOfficialDevQns, 
                               imgFile=config.testOfficialImgFeatures, 
                               config=config)
    
    model = AttentionModel(config)
    model.loadTrainedModel()
    model.runTest(testReader, config.testOfficialResultFile)
    model.destruct()

def runValTest():
    #Val set's split -- test
    print('Running Val Test')
    config = Attention_LapConfig(load=True)
    valTestReader = InputProcessor(config.testAnnotFile, 
                                 config.rawQnValTestFile, 
                                 config.testImgFile, 
                                 config,
                                 is_training=False)
    
    model = AttentionModel(config)
    model.loadTrainedModel()
    model.runPredict(valTestReader)
    model.destruct()
    

if __name__ == '__main__':
    loadOfficialTest()