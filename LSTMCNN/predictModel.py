'''
Created on 13 Feb 2018

@author: jwong

For CNN model
'''
from LSTMCNN_model import LSTMCNNModel
from CNN_GPUConfig import CNNGPUConfig
from CNN_LapConfig import CNNLapConfig
from InputProcessor import InputProcessor, TestProcessor
import argparse

        
def loadOfficialTest(args):
    if args.config == 'CPU':
        config = CNNLapConfig(load=True, args=args)
    elif args.config == 'GPU':
        config = CNNGPUConfig(load=True, args=args)
    
    testReader = TestProcessor(qnFile=config.testOfficialDevQns, 
                               imgFile=config.testOfficialImgPaths, 
                               config=config)
    
    model = LSTMCNNModel(config)
    model.loadTrainedModel(config.restoreModel, config.restoreModelPath)
    model.runTest(testReader, config.testOfficialResultFile)
    model.destruct()
    testReader.destruct()

def loadOfficialTestSpecs(args, restoreModel, restoreModelPath, config=None):
    if config is None:
        config = CNNGPUConfig(load=True, args=args)
    
    testReader = TestProcessor(qnFile=config.testOfficialDevQns, 
                               imgPathsFile=config.testOfficialImgPaths, 
                               config=config)
    
    model = LSTMCNNModel(config)
    model.loadTrainedModel(restoreModel, restoreModelPath)
    
    filename = 'toSubmit{}.json'.format(restoreModelPath.split('/')[-2])
    model.runTest(testReader, filename)
    model.destruct()
    return config

def runValTest(args):
    #Val set's split -- test
    print('Running Val Test')
    if args.config == 'CPU':
        config = CNNLapConfig(load=True, args=args)
    elif args.config == 'GPU':
        config = CNNGPUConfig(load=True, args=args)
    valTestReader = InputProcessor(config.testAnnotFile, 
                                 config.rawQnValTestFile, 
                                 config.valImgPaths, 
                                 config,
                                 is_training=False)
    
    model = LSTMCNNModel(config)
    model.loadTrainedModel(config.restoreModel, config.restoreModelPath)
    model.runPredict(valTestReader, config.csvResults)
    model.destruct()
    valTestReader.destruct()

def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--verbose', help='Display all print statement', 
                        action='store_true')
    parser.add_argument('-r', '--restorefile', help='Name of file to restore (.meta)')
    parser.add_argument('-p', '--restorepath', help='Name of path to file to restore')
    parser.add_argument('--att', choices=['qn', 'im'], default='qn')
    parser.add_argument('-a', '--action', choices=['otest', 'vtest', 'vis', 'solve'], default='vis')
    parser.add_argument('-s', '--seed', help='tf seed value', type=int)
    parser.add_argument('-c', '--config', choices=['GPU', 'CPU'], default='GPU', help='Name of path to file to restore')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parseArgs()
    restoreModel = 'results/CNN26Apr23-41/att26Apr23-41.meta'
    restoreModelPath = 'results/CNN26Apr23-41/'
    config = loadOfficialTestSpecs(args, restoreModel, restoreModelPath)
    
    restoreModel = 'results/CNN26Mar1-37/att26Mar1-37.meta'
    restoreModelPath = 'results/CNN26Mar1-37/'
    loadOfficialTestSpecs(args, restoreModel, restoreModelPath, config)
    