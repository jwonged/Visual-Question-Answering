'''
Created on 30 Mar 2018

@author: jwong
'''

from model.Image_AttModel import ImageAttentionModel
from model.Qn_AttModel import QnAttentionModel
from configs.Attention_LapConfig import Attention_LapConfig
from configs.Attention_GPUConfig import Attention_GPUConfig
from utils.TrainProcessors import TestProcessor, AttModelInputProcessor
import argparse

'''
1) Do Official Test
2) Internal Val Test
'''


def loadOfficialTest(args):
    #config = Attention_LapConfig(load=True, args)
    config = Attention_GPUConfig(load=True, args=args)
    
    testReader = TestProcessor(qnFile=config.testOfficialDevQns, 
                               imgFile=config.testOfficialImgFeatures, 
                               config=config)
    
    if args.att == 'qn':
        print('Attention over question and image model')
        model = QnAttentionModel(config)
    elif args.att == 'im':
        print('Attention over image model')
        model = ImageAttentionModel(config)
        
    model.loadTrainedModel(config.restoreModel, config.restoreModelPath)
    model.runTest(testReader, config.testOfficialResultFile)
    model.destruct()
    testReader.destruct()
    
def runValTest(args):
    #Val set's split -- test
    print('Running Val Test')
    config = Attention_LapConfig(load=True, args=args)
    
    valTestReader = TestProcessor(qnFile=config.valTestQns, 
                               imgFile=config.valImgFile, 
                               config=config)
    
    #valTestReader = TrainProcessors(config.testAnnotFile, 
    #                             config.rawQnValTestFile, 
    #                             config.valImgFile, 
    #                             config,
    #                             is_training=False)
    
    if args.att == 'qn':
        print('Attention over question and image model')
        model = QnAttentionModel(config)
    elif args.att == 'im':
        print('Attention over image model')
        model = ImageAttentionModel(config)
    model.loadTrainedModel(config.restoreQnImAttModel, 
                           config.restoreQnImAttModelPath)
    model.runTest(valTestReader, 'testResFile.json')
    model.destruct()
    valTestReader.destruct()

    
def internalValTest(args):
    import sys
    sys.path.insert(0, '/home/jwong/Documents/LinuxWorkspace/Visual-Question-Answering')
    from vqaTools.vqa import VQA
    from vqaTools.vqaEval import VQAEval
    
    config = Attention_LapConfig(load=False, args=args)
    annFile = config.originalAnnotVal
    quesFile = config.valTestQns
    resFile = 'testResFile.json'
    
    vqa = VQA(annFile, quesFile)
    vqaRes = vqa.loadRes(resFile, quesFile)
    
    vqaEval = VQAEval(vqa, vqaRes, n=2)
    vqaEval.evaluate() 
    
    # print accuracies
    print "\n"
    print "Overall Accuracy is: %.02f\n" %(vqaEval.accuracy['overall'])
    print "Per Question Type Accuracy is the following:"
    for quesType in vqaEval.accuracy['perQuestionType']:
        print "%s : %.02f" %(quesType, vqaEval.accuracy['perQuestionType'][quesType])
    print "\n"
    print "Per Answer Type Accuracy is the following:"
    for ansType in vqaEval.accuracy['perAnswerType']:
        print "%s : %.02f" %(ansType, vqaEval.accuracy['perAnswerType'][ansType])
    print "\n"


def predAnalysis(args):
    print('Running Val Test')
    predFile = 'Pred_QnAtt47.9.csv'
    config = Attention_GPUConfig(load=True, args=args)
    valTestReader = AttModelInputProcessor(config.testAnnotFile, 
                                 config.rawQnValTestFile, 
                                 config.valImgFile, 
                                 config,
                                 is_training=False)
    
    model = QnAttentionModel(config)
    model.loadTrainedModel(config.restoreModel, config.restoreModelPath)
    model.runPredict(valTestReader, predFile)
    model.destruct()
    valTestReader.destruct()

def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--restorefile', help='Name of file to restore (.meta)')
    parser.add_argument('-p', '--restorepath', help='Name of path to file to restore')
    parser.add_argument('--att', choices=['qn', 'im'], default='qn')
    parser.add_argument('--attfunc', choices=['sigmoid', 'softmax'], default='softmax')
    parser.add_argument('-a', '--action', choices=['otest', 'val'], default='val')
    parser.add_argument('-s', '--seed', help='tf seed value', type=int)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    pass