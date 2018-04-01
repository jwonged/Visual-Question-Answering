'''
Created on 15 Jan 2018

@author: jwong
'''
from LSTMIMGmodel import LSTMIMGmodel
from LSTMIMG_LapConfig import LSTMIMG_LapConfig
from LSTMIMG_GPUConfig import LSTMIMG_GPUConfig
from TrainProcessor import LSTMIMGProcessor
import argparse

def runtrain(args):
    #config = LSTMIMG_LapConfig(load=True)
    config = LSTMIMG_GPUConfig(load=True, args=args)
    
    trainReader = LSTMIMGProcessor(config.trainAnnotFile, 
                                 config.rawQnTrain, 
                                 config.trainImgFile, 
                                 config,
                                 is_training=True)
    
    valReader = LSTMIMGProcessor(config.valAnnotFile, 
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

def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--seed', help='tf seed value', type=int)
    parser.add_argument('-r', '--restorefile', help='Name of file to restore (.meta)')
    parser.add_argument('-p', '--restorepath', help='Name of path to file to restore')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parseArgs()
    runtrain(args)