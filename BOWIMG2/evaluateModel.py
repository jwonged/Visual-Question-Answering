'''
Created on 30 Mar 2018

@author: jwong
'''

from utils.BOWIMG_Processor import BOWIMGProcessor, TestProcessor
from configs.LaptopConfig import BOWIMG_LapConfig
from configs.GPUConfig import BOWIMG_GPUConfig
from model.BOWIMG_Model import BOWIMGModel
import argparse

'''
1) Do Official Test
2) Internal Val Test
'''

def loadOfficialTest(args):
    #config = BOWIMG_LapConfig(load=True, args)
    config = BOWIMG_GPUConfig(load=True, args=args)
    
    testReader = TestProcessor(qnFile=config.testOfficialDevQns, 
                               imgFile=config.testOfficialImgFeatures, 
                               config=config)
    
    print('Using BOWIMG Model')
    model = BOWIMGModel(config)
        
    model.loadTrainedModel(config.restoreModel, config.restoreModelPath)
    model.runTest(testReader, config.testOfficialResultFile)
    model.destruct()
    testReader.destruct()
    
def runValTest(args):
    #Val set's split -- test
    print('Running Val Test')
    config = BOWIMG_LapConfig(load=True, args=args)
    
    valTestReader = TestProcessor(qnFile=config.valTestQns, 
                               imgFile=config.valImgFile, 
                               config=config)
    
    #valTestReader = TrainProcessors(config.testAnnotFile, 
    #                             config.rawQnValTestFile, 
    #                             config.valImgFile, 
    #                             config,
    #                             is_training=False)
    
    print('Using BOWIMG Model')
    model = BOWIMGModel(config)
    
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
    
    config = BOWIMG_LapConfig(load=False, args=args)
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
    config = BOWIMG_GPUConfig(load=True, args=args)
    valTestReader = BOWIMGProcessor(config.testAnnotFile, 
                                 config.rawQnValTestFile, 
                                 config.valImgFile, 
                                 config,
                                 is_training=False)
    
    print('Using BOWIMG Model')
    model = BOWIMGModel(config)
    
    model.loadTrainedModel(config.restoreModel, config.restoreModelPath)
    model.runPredict(valTestReader, predFile)
    model.destruct()
    valTestReader.destruct()

def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--restorefile', help='Name of file to restore (.meta)')
    parser.add_argument('-p', '--restorepath', help='Name of path to file to restore')
    parser.add_argument('-a', '--action', choices=['otest', 'val'], default='val')
    parser.add_argument('-s', '--seed', help='tf seed value', type=int)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parseArgs()
    loadOfficialTest(args)