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
Read in annotations file and process input into X and Y batch to produce preprocessed files
clean = remove 0-vector answer classes
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
    
    def getNumOfAns(self):
        return self.numOfAns
        
    def readAnnotFile(self, annotFileName):
        #Read file, preprocess and append to xlabel, ylabel
        print('Reading ' + annotFileName)
        with open(annotFileName) as annotFile:
            annotBatch = json.load(annotFile)
        
        for annot in annotBatch:
            singleAns = self.resolveAnswer(annot['answers'])
            
            #encode ans if within ans classes
            if (singleAns not in self.ansClassMap):
                self.unclassifiedAns += 1
                continue
            self.ylabels.append(self.ansClassMap[singleAns])
            
            #encode qn and image
            qnVec = self.qnProcessor.getEncodedQn(annot['question_id'])
            imgVec = self.imgData[str(annot['image_id'])][0]
            xVec = qnVec + imgVec
            self.xlabels.append(xVec)

            #checks
            self.numOfAns += 1
            if(self.numOfAns%(len(annotBatch)/5) == 0):
                print('Number of ans processed: ' + str(self.numOfAns))

        print('Batch size produced: ' + str(self.numOfAns))

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
            index += 1 

        return ansVec, ansClassMap
    
    def getData(self):
        print ('Total number of Ans' + str(self.numOfAns))
        return self.xlabels, self.ylabels
    
def pickleData(dataX, dataY, xfile, yfile):
    print('Writing ' + xfile)
    with open(xfile, 'wb') as pklFile:
        pickle.dump(dataX, pklFile)
    print('Writing ' + yfile)
    with open(yfile, 'wb') as pklFile:
        pickle.dump(dataY, pklFile)

def main():
    mostFreqAnswersFile = '/home/jwong/Documents/LinuxWorkspace/Visual-Question-Answering/resources/1000MostFreqAnswers.csv'
    
    print('Loading files...')
    ###########-----------------------------------------Train------------------------
    xTrainPickle = '/media/jwong/Transcend/VQADataset/TrainSet/XYTrainData/sparseCleanWVsum1000Trainx'
    yTrainPickle = '/media/jwong/Transcend/VQADataset/TrainSet/XYTrainData/sparseCleanWVsum1000Trainy'
    
    trainWVQnsFile = '/media/jwong/Transcend/VQADataset/TrainSet/ExtractedQnFeatures/word2VecAddQnFeatures_Train.json'
    trainImgFile = '/media/jwong/Transcend/VQADataset/TrainSet/ExtractedImageFeatures/VQAImgFeatures_Train.json'
    trainProcessor = XYBatchProducer(trainWVQnsFile, trainImgFile, mostFreqAnswersFile)
    
    trainAnnotPath = '/media/jwong/Transcend/VQADataset/TrainSet/inputBatches/trainMiniBatches/TrainMiniBatch'
    
    for i in range(1,26):
        trainProcessor.readAnnotFile(trainAnnotPath+str(i)+'.json')
    
    dataX, dataY = trainProcessor.getData()
    third = len(dataX)/3
    
    pickleData(dataX[:third], dataY[:third], xTrainPickle+'1.pkl', yTrainPickle+'1.pkl')
    pickleData(dataX[third:2*third], dataY[third:2*third], xTrainPickle+'2.pkl', yTrainPickle+'2.pkl')
    pickleData(dataX[2*third:], dataY[2*third:], xTrainPickle+'3.pkl', yTrainPickle+'3.pkl')
    pickleData(dataX, dataY, xTrainPickle+'All.pkl', yTrainPickle+'All.pkl')
        
    
    print('Train files completed. Each batch file has ' + str(third))
    print('Unclassified train answers= ' + str(trainProcessor.unclassifiedAns))
    del trainProcessor
    del dataX[:]
    del dataY[:]
    
    
    ###########-----------------------------------------Val------------------------
    
    xValPickle = '/media/jwong/Transcend/VQADataset/ValTestSet/XYValTestData/sparseCleanWVsum1000valx.pkl'
    yValPickle = '/media/jwong/Transcend/VQADataset/ValTestSet/XYValTestData/sparseCleanWVsum1000valy.pkl'
    
    
    valTestWVQnsFile = '/media/jwong/Transcend/VQADataset/ValTestSet/ExtractedQnFeatures/word2VecAddQnFeatures_valTest.json'
    valImgFile = '/media/jwong/Transcend/VQADataset/ValTestSet/VQAImgFeatures_Val.json'
    
    valAnnotPath = '/media/jwong/Transcend/VQADataset/ValTestSet/valMiniBatches/valMiniBatch'
    
    valProcessor = XYBatchProducer(valTestWVQnsFile, valImgFile, mostFreqAnswersFile)
    
    for i in range(1,8):
        valProcessor.readAnnotFile(valAnnotPath+str(i)+'.json')
        
    dataX, dataY = valProcessor.getData()
    pickleData(dataX, dataY, xValPickle, yValPickle)
    print('Val files completed for 1000.')
    del valProcessor
    del dataX[:]
    del dataY[:]
    
    ###########-----------------------------------------test------------------------
    valTestWVQnsFile = '/media/jwong/Transcend/VQADataset/ValTestSet/ExtractedQnFeatures/word2VecAddQnFeatures_valTest.json'    

    xTestPickle = '/media/jwong/Transcend/VQADataset/ValTestSet/XYValTestData/sparseCleanWVsum1000testx.pkl'
    yTestPickle = '/media/jwong/Transcend/VQADataset/ValTestSet/XYValTestData/sparseCleanWVsum1000testy.pkl'
    
    testImgFile = '/media/jwong/Transcend/VQADataset/ValTestSet/VQAImgFeatures_Test.json'
    
    testAnnotPath = '/media/jwong/Transcend/VQADataset/ValTestSet/testMiniBatches/testMiniBatch'
    testProcessor = XYBatchProducer(valTestWVQnsFile, testImgFile, mostFreqAnswersFile)
    
    for i in range(1,8):
        testProcessor.readAnnotFile(testAnnotPath+str(i)+'.json')
    
    dataX, dataY = testProcessor.getData()
    pickleData(dataX, dataY, xTestPickle, yTestPickle)
    del testProcessor
    del dataX[:]
    del dataY[:]
    print('Test files Completed for 1000.')

def main2():
    #clean sets with all train answers
    mostFreqAnswersFile = '/home/jwong/Documents/LinuxWorkspace/Visual-Question-Answering/resources/allTrainAnswers.csv'
    
    print('Loading files for all train...')
    ###########-----------------------------------------Train------------------------
    xTrainPickle = '/media/jwong/Transcend/VQADataset/TrainSet/XYTrainData/sparseCleanWVsumAllAnsTrainx'
    yTrainPickle = '/media/jwong/Transcend/VQADataset/TrainSet/XYTrainData/sparseCleanWVsumAllAnsTrainy'
    
    trainWVQnsFile = '/media/jwong/Transcend/VQADataset/TrainSet/ExtractedQnFeatures/word2VecAddQnFeatures_Train.json'
    trainImgFile = '/media/jwong/Transcend/VQADataset/TrainSet/ExtractedImageFeatures/VQAImgFeatures_Train.json'
    trainProcessor = XYBatchProducer(trainWVQnsFile, trainImgFile, mostFreqAnswersFile)
    
    trainAnnotPath = '/media/jwong/Transcend/VQADataset/TrainSet/inputBatches/trainMiniBatches/TrainMiniBatch'
    
    for i in range(1,26):
        trainProcessor.readAnnotFile(trainAnnotPath+str(i)+'.json')
    
    dataX, dataY = trainProcessor.getData()
    third = len(dataX)/3
    
    pickleData(dataX[:third], dataY[:third], xTrainPickle+'1.pkl', yTrainPickle+'1.pkl')
    pickleData(dataX[third:2*third], dataY[third:2*third], xTrainPickle+'2.pkl', yTrainPickle+'2.pkl')
    pickleData(dataX[2*third:], dataY[2*third:], xTrainPickle+'3.pkl', yTrainPickle+'3.pkl')
    pickleData(dataX, dataY, xTrainPickle+'All.pkl', yTrainPickle+'All.pkl')
    
    print('Train files completed(AllAns).')
    print('Unclassified train answers= ' + str(trainProcessor.unclassifiedAns))
    del trainProcessor
    del dataX[:]
    del dataY[:]
    
    
    ###########-----------------------------------------Val------------------------
    
    xValPickle = '/media/jwong/Transcend/VQADataset/ValTestSet/XYValTestData/sparseCleanWVsumAllAnsvalx.pkl'
    yValPickle = '/media/jwong/Transcend/VQADataset/ValTestSet/XYValTestData/sparseCleanWVsumAllAnsvaly.pkl'
    
    
    valTestWVQnsFile = '/media/jwong/Transcend/VQADataset/ValTestSet/ExtractedQnFeatures/word2VecAddQnFeatures_valTest.json'
    valImgFile = '/media/jwong/Transcend/VQADataset/ValTestSet/VQAImgFeatures_Val.json'
    
    valAnnotPath = '/media/jwong/Transcend/VQADataset/ValTestSet/valMiniBatches/valMiniBatch'
    
    valProcessor = XYBatchProducer(valTestWVQnsFile, valImgFile, mostFreqAnswersFile)
    
    for i in range(1,8):
        valProcessor.readAnnotFile(valAnnotPath+str(i)+'.json')
        
    dataX, dataY = valProcessor.getData()
    pickleData(dataX, dataY, xValPickle, yValPickle)
    print('Val files completed (AllAns).')
    del valProcessor
    del dataX[:]
    del dataY[:]
    
    ###########-----------------------------------------test------------------------
    valTestWVQnsFile = '/media/jwong/Transcend/VQADataset/ValTestSet/ExtractedQnFeatures/word2VecAddQnFeatures_valTest.json'    

    xTestPickle = '/media/jwong/Transcend/VQADataset/ValTestSet/XYValTestData/sparseCleanWVsumAllAnstestx.pkl'
    yTestPickle = '/media/jwong/Transcend/VQADataset/ValTestSet/XYValTestData/sparseCleanWVsumAllAnstesty.pkl'
    
    testImgFile = '/media/jwong/Transcend/VQADataset/ValTestSet/VQAImgFeatures_Test.json'
    
    testAnnotPath = '/media/jwong/Transcend/VQADataset/ValTestSet/testMiniBatches/testMiniBatch'
    testProcessor = XYBatchProducer(valTestWVQnsFile, testImgFile, mostFreqAnswersFile)
    
    for i in range(1,8):
        testProcessor.readAnnotFile(testAnnotPath+str(i)+'.json')
    
    dataX, dataY = testProcessor.getData()
    pickleData(dataX, dataY, xTestPickle, yTestPickle)
    
    print('Test files Completed(AllAns).')

if __name__ == "__main__":
    main()
    print('First main fully Completed.')
    main2()
    print('Second main fully Completed.')
    
