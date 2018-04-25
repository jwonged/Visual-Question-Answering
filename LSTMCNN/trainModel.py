'''
Created on 15 Jan 2018

@author: jwong

For LSTMCNN model
'''
from LSTMCNN_model import LSTMCNNModel
from CNN_GPUConfig import CNNGPUConfig
from CNN_LapConfig import CNNLapConfig
from InputProcessor import InputProcessor
import argparse

def runtrain(args):
    if args.config == 'CPU':
        config = CNNLapConfig(load=True, args=args)
    elif args.config == 'GPU':
        config = CNNGPUConfig(load=True, args=args)
    
    trainReader = InputProcessor(config.trainAnnotFile, 
                                 config.rawQnTrain, 
                                 config.trainImgPaths, 
                                 config,
                                 is_training=True)
    
    valReader = InputProcessor(config.valAnnotFile, 
                                 config.rawQnValTestFile, 
                                 config.valImgPaths, 
                                 config,
                                 is_training=False)
    
    model = LSTMCNNModel(config)

    model.construct()
    model.train(trainReader, valReader, config.logFile)
    model.destruct()
    return config
    
def loadOfficialTest(args, restoreModel=None, restoreModelPath=None):
    if args.config == 'CPU':
        config = CNNLapConfig(load=True, args=args)
    elif args.config == 'GPU':
        config = CNNGPUConfig(load=True, args=args)
    
    testReader = TestProcessor(qnFile=config.testOfficialDevQns, 
                               imgFile=config.testOfficialImgPaths, 
                               config=config)
    
    model = LSTMCNNModel(config)
    model.loadTrainedModel(restoreModel, restoreModelPath)
    model.runTest(testReader, '25AprLSTMCNNSubmission.json')
    model.destruct()
    testReader.destruct()

def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--seed', help='tf seed value', type=int)
    parser.add_argument('-v', '--verbose', help='Display all print statement', action='store_true')
    parser.add_argument('--att', choices=['qn', 'im', 'none'], default='None')
    parser.add_argument('-r', '--restorefile', help='Name of file to restore (.meta)')
    parser.add_argument('-p', '--restorepath', help='Name of path to file to restore')
    parser.add_argument('-c', '--config', choices=['GPU', 'CPU'], default='GPU', help='Name of path to file to restore')
    args = parser.parse_args()
    return args
    
if __name__ == '__main__':
    args = parseArgs()
    config = runtrain(args)
    loadOfficialTest(args, 
                     restoreModel=config.saveModelFile+'.meta', 
                     restoreModelPath=config.saveModelPath)
