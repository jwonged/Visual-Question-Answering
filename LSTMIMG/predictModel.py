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
                               config)
    
    model = LSTMIMGmodel(config)
    model.loadTrainedModel()
    model.runPredict(testReader)
    model.destruct()


if __name__ == '__main__':
    loadAndTest()