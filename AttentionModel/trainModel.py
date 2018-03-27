'''
Created on 15 Jan 2018

@author: jwong
'''
from Image_AttModel import ImageAttentionModel
from Qn_AttModel import QnAttentionModel
from Attention_LapConfig import Attention_LapConfig
from Attention_GPUConfig import Attention_GPUConfig
from InputProcessor import InputProcessor
import argparse

def runtrain(args):
    #config = Attention_LapConfig(load=True, args)
    config = Attention_GPUConfig(load=True, args=args)
    #self.attentionType = 'img'
    
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

def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--seed', help='tf seed value', type=int)
    parser.add_argument('-v', '--verbose', help='Display all print statement', action='store_true')
    parser.add_argument('--att', choices=['qn', 'im'], default='qn')
    parser.add_argument('--attfunc', choices=['sigmoid', 'softmax'], default='softmax')
    parser.add_argument('-r', '--restorefile', help='Name of file to restore (.meta)')
    parser.add_argument('-p', '--restorepath', help='Name of path to file to restore')
    args = parser.parse_args()
    return args
    
if __name__ == '__main__':
    args = parseArgs()
    runtrain(args)