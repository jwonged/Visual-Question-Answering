'''
Created on 30 Mar 2018

@author: jwong
'''

from utils.BOWIMG_Processor import BOWIMGProcessor, TestProcessor
from configs.LaptopConfig import BOWIMG_LapConfig
from configs.GPUConfig import BOWIMG_GPUConfig
from model.BOWIMG_Model import BOWIMGModel
import argparse
import csv
import random

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

def validateInternalTestSet(args):
    from vqaTools.vqaInternal import VQA
    from vqaTools.vqaEval import VQAEval
    
    #config = Attention_LapConfig(load=True, args)
    config = BOWIMG_GPUConfig(load=True, args=args)
    
    restoreModel = config.restoreModel
    restoreModelPath = config.restoreModelPath
    
    print('Running Validation Test on Model')
    valTestReader = BOWIMGProcessor(config.testAnnotFile, 
                                 config.rawQnValTestFile, 
                                 config.testImgFile, 
                                 config,
                                 is_training=False)
    
    print('Using BOWIMG model')
    model = BOWIMGModel(config)
    
    model.loadTrainedModel(restoreModel, restoreModelPath)
    predFile = '{}PredsBOW.csv'.format(restoreModelPath)
    results, strictAcc = model.runPredict(valTestReader, predFile)
    model.destruct()
    valTestReader.destruct()
    print('predictions made')
    
    vqa = VQA(config.testAnnotFileUnresolved, config.originalValQns)
    vqaRes = vqa.loadRes(results, config.originalValQns)
    vqaEval = VQAEval(vqa, vqaRes, n=2)
    vqaEval.evaluate() 
    
    print('Writing to file..')
    writeToFile(vqaEval, restoreModelPath, vqa, vqaRes, strictAcc)
        
def writeToFile(vqaEval, restoreModelPath, vqa, vqaRes, strictAcc):
    outputFile = '{}resultbrkdwnBOW.csv'.format(restoreModelPath)
    with open(outputFile, 'wb') as csvfile:
        logWriter = csv.writer(csvfile)
        logWriter.writerow(['StrictAcc: {}'.format(strictAcc)])
        msg = "Overall Accuracy is: %.02f\n" %(vqaEval.accuracy['overall'])
        logWriter.writerow([msg])
        print(msg)
        
        #qn type breakdown
        msg = "Per Question Type Accuracy is the following:"
        logWriter.writerow([msg])
        print(msg)
        for quesType in vqaEval.accuracy['perQuestionType']:
            msg = "%s : %.02f" %(quesType, vqaEval.accuracy['perQuestionType'][quesType])
            print(msg)
            logWriter.writerow([msg])
        
        #answer type breakdown
        msg = "Per Answer Type Accuracy is the following:"
        print(msg)
        logWriter.writerow([msg])
        for ansType in vqaEval.accuracy['perAnswerType']:
            msg = "%s : %.02f" %(ansType, vqaEval.accuracy['perAnswerType'][ansType])
            logWriter.writerow([msg])
            print(msg)
        
        #Retrieve random low score answer
        evals = [quesId for quesId in vqaEval.evalQA if vqaEval.evalQA[quesId]<35]   #35 is per question percentage accuracy
        if len(evals) > 0:
            print 'ground truth answers'
            randomEval = random.choice(evals)
            randomAnn = vqa.loadQA(randomEval)
            qns, answers = vqa.showQA(randomAnn)
            print(type(randomEval))
            img_id = vqa.getImgFromQnId(randomEval)
            logWriter.writerow(['Retrieving low scoring qns'])
            logWriter.writerow(['Img:']+[img_id])
            print(img_id)
            logWriter.writerow(['qn:']+qns)
            print(qns)
            logWriter.writerow(['answers:']+answers)
            print(answers)
            msg = 'generated answer (accuracy %.02f)'%(vqaEval.evalQA[randomEval])
            print(msg)
            logWriter.writerow([msg])
            ann = vqaRes.loadQA(randomEval)[0]
            msg = "Answer:   %s\n" %(ann['answer'])
            logWriter.writerow([msg])
            print(msg)
            
    print('Written to {}'.format(outputFile))
    
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
    if args.action == 'otest':
        loadOfficialTest(args)
    elif args.action == 'val':
        validateInternalTestSet(args)