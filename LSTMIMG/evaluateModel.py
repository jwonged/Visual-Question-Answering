'''
Created on 13 Feb 2018

@author: jwong
'''
from LSTMIMGmodel import LSTMIMGmodel
from LSTMIMG_LapConfig import LSTMIMG_LapConfig
from LSTMIMG_GPUConfig import LSTMIMG_GPUConfig
from TrainProcessor import LSTMIMGProcessor, TestProcessor
import argparse
import random
import pickle
import csv


def loadOfficialTest(args):
    config = LSTMIMG_GPUConfig(load=True, args=args)
    
    testReader = TestProcessor(qnFile=config.testOfficialDevQns, 
                               imgFile=config.testOfficialImgFeatures, 
                               config=config)
    
    model = LSTMIMGmodel(config)
    model.loadTrainedModel(config.restoreModel, config.restoreModelPath)
    model.runTest(testReader, config.testOfficialResultFile)
    model.destruct()

def validateInternalTestSet(args):
    from vqaTools.vqaInternal import VQA
    from vqaTools.vqaEval import VQAEval
    
    #config = LSTMIMG_LapConfig(load=True, args)
    config = LSTMIMG_GPUConfig(load=True, args=args)
    
    restoreModel = config.restoreModel
    restoreModelPath = config.restoreModelPath
    
    print('Running Validation Test on Model')
    valTestReader = LSTMIMGProcessor(config.testAnnotFile, 
                                 config.rawQnValTestFile, 
                                 config.testImgFile, 
                                 config,
                                 is_training=False)
    
    model = LSTMIMGmodel(config)
    
    model.loadTrainedModel(restoreModel, restoreModelPath)
    predFile = '{}PredsLSTMEverything.csv'.format(restoreModelPath)
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
    outputFile = '{}resultbrkdwnLSTM.csv'.format(restoreModelPath)
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