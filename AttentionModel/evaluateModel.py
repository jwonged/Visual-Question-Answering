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
import csv

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

def validateInternalTestSet(args):
    from vqaTools.vqaInternal import VQA
    from vqaTools.vqaEval import VQAEval
    
    #config = Attention_LapConfig(load=True, args)
    config = Attention_GPUConfig(load=True, args=args)
    
    restoreModel = config.restoreModel
    restoreModelPath = config.restoreModelPath
    
    vqa = VQA(config.testAnnotFileUnresolved, config.originalValQns)
    
    print('Running Validation Test on Model')
    valTestReader = AttModelInputProcessor(config.testAnnotFile, 
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
    
    model.loadTrainedModel(restoreModel, restoreModelPath)
    res, strictAcc = model.runPredict(
        valTestReader, '{}PredsAtt{}.csv'.format(restoreModelPath, args.att))
    model.destruct()
    valTestReader.destruct()
    
    
    vqaRes = vqa.loadRes(res, config.originalValQns)
    vqaEval = VQAEval(vqa, vqaRes, n=2)
    vqaEval.evaluate() 
    
    writeToFile(vqaEval, restoreModelPath, vqa, vqaRes, args, strictAcc)
        
def writeToFile(vqaEval, restoreModelPath, vqa, vqaRes, args, strictAcc):
    outputFile = '{}resultbrkdwnAtt{}.csv'.format(restoreModelPath, args.att)
    with open(outputFile, 'wb') as csvfile:
        csvfile.writerow['StrictAcc: {}'.format(strictAcc)]
        msg = "Overall Accuracy is: %.02f\n" %(vqaEval.accuracy['overall'])
        csvfile.writerow([msg])
        print(msg)
        msg = "Per Question Type Accuracy is the following:"
        csvfile.writerow([msg])
        print(msg)
        
        for ansType in vqaEval.accuracy['perAnswerType']:
            msg = "%s : %.02f" %(ansType, vqaEval.accuracy['perAnswerType'][ansType])
            csvfile.writerow([msg])
            print(msg)
        
        # demo how to use evalQA to retrieve low score result
        evals = [quesId for quesId in vqaEval.evalQA if vqaEval.evalQA[quesId]<35]   #35 is per question percentage accuracy
        if len(evals) > 0:
            print 'ground truth answers'
            randomEval = random.choice(evals)
            randomAnn = vqa.loadQA(randomEval)
            qns, answers = vqa.showQA(randomAnn)
            img_ids = vqa.getImgIds(quesIds=[randomEval])
            csvfile.writerow(['Retrieving low scoring qns'])
            csvfile.writerow(['Img:']+img_ids)
            csvfile.writerow(['qn:']+qns)
            csvfile.writerow(['answers:']+answers)
            msg = 'generated answer (accuracy %.02f)'%(vqaEval.evalQA[randomEval])
            print(msg)
            csvfile.writerow([msg])
            ann = vqaRes.loadQA(randomEval)[0]
            msg = "Answer:   %s\n" %(ann['answer'])
            csvfile.writerow([msg])
            
    print('Written to {}'.format(outputFile))
    
    
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

import matplotlib.pyplot as plt
import skimage.io as io
import json
import random
import os
def internalValTest(args):
    import sys
    #sys.path.insert(0, '/home/jwong/Documents/LinuxWorkspace/Visual-Question-Answering')
    from vqaTools.vqaInternal import VQA
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
    
    # demo how to use evalQA to retrieve low score result
    evals = [quesId for quesId in vqaEval.evalQA if vqaEval.evalQA[quesId]<35]   #35 is per question percentage accuracy
    if len(evals) > 0:
        print('ground truth answers')
        randomEval = random.choice(evals) #
        print('RandomEval {}'.format(randomEval))
        randomAnn = vqa.loadQA(randomEval) 
        qns, answers = vqa.showQA(randomAnn) 
        print(qns)
        print(answers)
        img_ids = vqa.getImgIds(quesIds=[randomEval])
        print(img_ids)
    
        print '\n'
        print 'generated answer (accuracy %.02f)'%(vqaEval.evalQA[randomEval])
        ann = vqaRes.loadQA(randomEval)[0]
        print "Answer:   %s\n" %(ann['answer'])
    
        #imgId = randomAnn[0]['image_id']
        #imgFilename = 'COCO_' + dataSubType + '_'+ str(imgId).zfill(12) + '.jpg'
        #if os.path.isfile(imgDir + imgFilename):
        #    I = io.imread(imgDir + imgFilename)
        #    plt.imshow(I)
        #    plt.axis('off')
        #    plt.show()
    
    # plot accuracy for various question types
    plt.bar(range(len(vqaEval.accuracy['perQuestionType'])), vqaEval.accuracy['perQuestionType'].values(), align='center')
    plt.xticks(range(len(vqaEval.accuracy['perQuestionType'])), vqaEval.accuracy['perQuestionType'].keys(), rotation='0',fontsize=10)
    plt.title('Per Question Type Accuracy', fontsize=10)
    plt.xlabel('Question Types', fontsize=10)
    plt.ylabel('Accuracy', fontsize=10)
    plt.show()


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
    args = parseArgs()
    if args.action == 'otest':
        loadOfficialTest(args)
    elif args.action == 'val':
        validateInternalTestSet(args)
        