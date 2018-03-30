'''
Created on 15 Jan 2018

@author: jwong
'''

from model.BOWIMG_Model import BOWIMGModel
from configs.LaptopConfig import BOWIMG_LapConfig
from configs.GPUConfig import BOWIMG_GPUConfig
from utils.BOWIMG_Processor import BOWIMGProcessor
import argparse

#import sys
#sys.path.insert(0, '/home/jwong/Documents/LinuxWorkspace/Visual-Question-Answering')

def runtrain(args):
    #config = BOWIMG_LapConfig(load=True, args)
    config = BOWIMG_GPUConfig(load=True, args=args)
    
    trainReader = BOWIMGProcessor(config.trainAnnotFile, 
                                 config.rawQnTrain, 
                                 config.trainImgFile, 
                                 config,
                                 is_training=True)
    
    valReader = BOWIMGProcessor(config.valAnnotFile, 
                                 config.rawQnValTestFile, 
                                 config.valImgFile, 
                                 config,
                                 is_training=False)
    
    print('Using BOWIMG model')
    model = BOWIMGModel(config)

    model.construct()
    model.train(trainReader, valReader, config.logFile)
    model.destruct()
    trainReader.destruct()
    valReader.destruct()

def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--seed', help='tf seed value', type=int)
    parser.add_argument('-v', '--verbose', help='Display all print statement', action='store_true')
    parser.add_argument('-r', '--restorefile', help='Name of file to restore (.meta)')
    parser.add_argument('-p', '--restorepath', help='Name of path to file to restore')
    args = parser.parse_args()
    return args
    
if __name__ == '__main__':
    args = parseArgs()
    runtrain(args)