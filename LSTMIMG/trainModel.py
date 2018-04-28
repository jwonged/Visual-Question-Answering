'''
Created on 15 Jan 2018

@author: jwong
'''
from LSTMIMGmodel import LSTMIMGmodel
from LSTMIMG_LapConfig import LSTMIMG_LapConfig
from LSTMIMG_GPUConfig import LSTMIMG_GPUConfig
from TrainProcessor import LSTMIMGProcessor
from evaluateModel import loadOfficialTest, validateInternalTestSet
from model_utils import DummyReader
import argparse

def runtrain(args):
    #config = LSTMIMG_LapConfig(load=True, args=args)
    
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
    model.train(trainReader, valReader, config.logFile)
    #model.train(dumReader, dumReader,'dumlog.csv')
    model.destruct()
    
    return config

def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--seed', help='tf seed value', type=int)
    parser.add_argument('-r', '--restorefile', help='Name of file to restore (.meta)')
    parser.add_argument('-p', '--restorepath', help='Name of path to file to restore')
    parser.add_argument('--lstmtype', choices=['bi', 'uni'], default='bi')
    parser.add_argument('--useuntrainedembed', help='use untrained embeddings', action='store_false')
    parser.add_argument('--donttrainembed', help='use untrained embeddings', action='store_false')
    parser.add_argument('--useconcat', help='use concat instead of elmult', action='store_false')
    parser.add_argument('--noshuffle', help='Do not shuffle dataset', action='store_false')
    parser.add_argument('--mmAtt', help='Use multimodal attention', action='store_true')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parseArgs()
    config = runtrain(args)
    loadOfficialTest(args, config.saveModelFile+'.meta', config.saveModelPath)
    validateInternalTestSet(args, config.saveModelFile+'.meta', config.saveModelPath)