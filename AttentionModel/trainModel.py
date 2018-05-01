'''
Created on 15 Jan 2018

@author: jwong
'''
from model.Image_AttModel import ImageAttentionModel
from model.Qn_AttModel import QnAttentionModel
from configs.Attention_LapConfig import Attention_LapConfig
from configs.Attention_GPUConfig import Attention_GPUConfig
from utils.TrainProcessors import AttModelInputProcessor
from evaluateModel import loadOfficialTest, validateInternalTestSet
import argparse

def runtrain(args):
    #config = Attention_LapConfig(load=True, args)
    config = Attention_GPUConfig(load=True, args=args)
    
    trainReader = AttModelInputProcessor(config.trainAnnotFile, 
                                 config.rawQnTrain, 
                                 config.trainImgFile, 
                                 config,
                                 is_training=True)
    
    valReader = AttModelInputProcessor(config.valAnnotFile, 
                                 config.rawQnValTestFile, 
                                 config.valImgFile, 
                                 config,
                                 is_training=False)
     
    if args.att == 'qn':
        print('Attention over question and image model')
        model = QnAttentionModel(config)
    elif args.att == 'im':
        print('Attention over image model')
        model = ImageAttentionModel(config)

    model.construct()
    model.train(trainReader, valReader, config.logFile)
    model.destruct()
    trainReader.destruct()
    valReader.destruct()
    
    return config

def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--seed', help='tf seed value', type=int)
    parser.add_argument('-v', '--verbose', help='Display all print statement', action='store_true')
    parser.add_argument('--att', choices=['qn', 'im'], default='qn')
    parser.add_argument('--attfunc', choices=['sigmoid', 'softmax'], default='softmax')
    parser.add_argument('-r', '--restorefile', help='Name of file to restore (.meta)')
    parser.add_argument('-p', '--restorepath', help='Name of path to file to restore')
    parser.add_argument('--notopk', help='No loading topk', action='store_true')
    parser.add_argument('--noqnatt', help='No loading qnAtt', action='store_true')
    parser.add_argument('--debugmode', help='Trace printing', action='store_true')
    parser.add_argument('--stackAtt', help='Use stacked attention', action='store_true')
    parser.add_argument('--attComb', choices=['concat', 'mult', 'add'], default='concat')
    parser.add_argument('--qnAttf', choices=['softmax', 'sigmoid'], default='softmax')
    parser.add_argument('--mmAtt', help='Use multimodal attention', action='store_true')
    args = parser.parse_args()
    return args
    
if __name__ == '__main__':
    args = parseArgs()
    config = runtrain(args)
    model = loadOfficialTest(args, 
                     restoreModel=config.saveModelFile+'.meta', 
                     restoreModelPath=config.saveModelPath)
    validateInternalTestSet(args, model, config.saveModelPath)
    
    
    