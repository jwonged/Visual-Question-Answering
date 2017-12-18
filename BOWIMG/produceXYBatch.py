'''
Created on 17 Dec 2017

@author: jwong
'''
import json
import csv
from collections import Counter
import pickle
from WVSumQuestionProcessor import WVSumQuestionProcessor

'''
Read in annotations file and process input into X and Y batch
'''

class XYBatchProducer:
    def __init__(self, questionFile, imageFile, mostFreqAnswersFile):
        print('Reading ' + imageFile)
        with open(imageFile) as jFile:
            self.imgData = json.load(jFile)
        print('Reading ' + questionFile)
        self.qnProcessor = WVSumQuestionProcessor(questionFile)
        self.mostFreqAnswersFile = mostFreqAnswersFile
        print('Reading ' + mostFreqAnswersFile)
        self.ansClasses, self.ansClassMap = self.getNMostFreqAnswers()
        self.ylabels = []
        self.xlabels = []
        self.numOfAns = 0
        self.ansClassLen = len(self.ansClasses)
        self.unclassifiedAns = 0
        
    def readAnnotFile(self, annotFileName):
        print('Reading ' + annotFileName)
        with open(annotFileName) as annotFile:
            annotBatch = json.load(annotFile)
        
        
        for annot in annotBatch:
            singleAns = self.resolveAnswer(annot['answers'])
            ansVec = self.encodeAns(singleAns, self.ansClassMap, self.ansClassLen)
            self.ylabels.append(ansVec)

            qnVec = self.qnProcessor.getEncodedQn(annot['question_id'])
            imgVec = self.imgData[str(annot['image_id'])][0]
            xVec = qnVec + imgVec
            self.xlabels.append(xVec)

            #checks
            self.numOfAns += 1
            if(self.numOfAns%(len(annotBatch)/20) == 0):
                print('Number of ans processed: ' + str(self.numOfAns))

        print('Batch size produced: ' + str(self.numOfAns))

    def encodeAns(self, ans, ansClassMap, ansClassLen):
        ansVec = [0] * ansClassLen
        if (ans in ansClassMap):
            ansVec[ansClassMap[ans]] = 1
        else:
            self.unclassifiedAns += 1
        return ansVec

    def resolveAnswer(self, possibleAnswersList):
        answers = []
        for answerDetails in possibleAnswersList:
            answers.append(answerDetails['answer'])
        mostCommon = Counter(answers).most_common(1)
        return mostCommon[0][0]

    def getNMostFreqAnswers(self):
        with open(self.mostFreqAnswersFile, 'rb') as ansFile:
            reader = csv.reader(ansFile, delimiter=',')
            ansVec = next(reader)

        index = 0
        ansClassMap = {}
        for word in ansVec:
            ansClassMap[word] = index
            index = index + 1 

        return ansVec, ansClassMap

    def pickleData(self, xfile, yfile):
        with open(xfile, 'wb') as pklFile:
            pickle.dump(self.xlabels, pklFile)
        with open(yfile, 'wb') as pklFile:
            pickle.dump(self.ylabels, pklFile)

if __name__ == "__main__":
    
    mostFreqAnswersFile = '/home/jwong/Documents/LinuxWorkspace/Visual-Question-Answering/resources/1000MostFreqAnswers.csv'
    
    print('Loading files...')
    '''
    ###########-----------------------------------------Train------------------------
    xTrainPickle = '/media/jwong/Transcend/VQADataset/TrainSet/XYTrainData/WVsum1000Trainx.pkl'
    yTrainPickle = '/media/jwong/Transcend/VQADataset/TrainSet/XYTrainData/WVsum1000Trainy.pkl'
    
    trainWVQnsFile = '/media/jwong/Transcend/VQADataset/TrainSet/ExtractedQnFeatures/word2VecAddQnFeatures_Train.json'
    trainImgFile = '/media/jwong/Transcend/VQADataset/TrainSet/ExtractedImageFeatures/VQAImgFeatures_Train.json'
    trainProcessor = XYBatchProducer(trainWVQnsFile, trainImgFile, mostFreqAnswersFile)
    
    trainAnnotPath = '/media/jwong/Transcend/VQADataset/TrainSet/inputBatches/trainMiniBatches/TrainMiniBatch'
    
    for i in range(1,26):
        trainProcessor.readAnnotFile(trainAnnotPath+str(i)+'.json')
        
    trainProcessor.pickleData(xfile=xTrainPickle, yfile=yTrainPickle)
    print('Train files completed.')
    print('Unclassified train answers= ' + str(trainProcessor.unclassifiedAns))
    
    ###########-----------------------------------------Val------------------------
    
    xValPickle = '/media/jwong/Transcend/VQADataset/ValTestSet/XYValTestData/WVsum1000valx.pkl'
    yValPickle = '/media/jwong/Transcend/VQADataset/ValTestSet/XYValTestData/WVsum1000valy.pkl'
    
    
    valTestWVQnsFile = '/media/jwong/Transcend/VQADataset/ValTestSet/ExtractedQnFeatures/word2VecAddQnFeatures_valTest.json'
    valImgFile = '/media/jwong/Transcend/VQADataset/ValTestSet/VQAImgFeatures_Val.json'
    
    valAnnotPath = '/media/jwong/Transcend/VQADataset/ValTestSet/valMiniBatches/valMiniBatch'
    
    valProcessor = XYBatchProducer(valTestWVQnsFile, valImgFile, mostFreqAnswersFile)
    
    for i in range(1,8):
        valProcessor.readAnnotFile(valAnnotPath+str(i)+'.json')
        
    
    valProcessor.pickleData(xfile=xValPickle, yfile=yValPickle)
    print('Val files completed.')
    
    ###########-----------------------------------------test------------------------
    valTestWVQnsFile = '/media/jwong/Transcend/VQADataset/ValTestSet/ExtractedQnFeatures/word2VecAddQnFeatures_valTest.json'
    
    xTestPickle = '/media/jwong/Transcend/VQADataset/ValTestSet/XYValTestData/WVsum1000testx.pkl'
    yTestPickle = '/media/jwong/Transcend/VQADataset/ValTestSet/XYValTestData/WVsum1000testy.pkl'
    
    testImgFile = '/media/jwong/Transcend/VQADataset/ValTestSet/VQAImgFeatures_Test.json'
    
    testAnnotPath = '/media/jwong/Transcend/VQADataset/ValTestSet/testMiniBatches/testMiniBatch'
    testProcessor = XYBatchProducer(valTestWVQnsFile, testImgFile, mostFreqAnswersFile)
    
    for i in range(1,8):
        testProcessor.readAnnotFile(testAnnotPath+str(i)+'.json')
    
    testProcessor.pickleData(xfile=xTestPickle, yfile=yTestPickle)
    '''
    
    
    
