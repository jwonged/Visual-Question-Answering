'''
Created on 13 Feb 2018

@author: jwong
'''
from LSTMIMGmodel import LSTMIMGmodel
from LSTMIMG_LapConfig import LSTMIMG_LapConfig
from LSTMIMG_GPUConfig import LSTMIMG_GPUConfig
from InputProcessor import InputProcessor
import pickle
import csv


def loadAndTest():
    config = LSTMIMG_LapConfig(load=True)
    #config = LSTMIMG_GPUConfig(load=True)
    
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
    
    
    model = LSTMIMGmodel(config)
    model.loadTrainedModel()
    model.runPredict(valReader)
    model.destruct()


if __name__ == '__main__':
    loadAndTest()