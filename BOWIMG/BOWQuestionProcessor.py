'''
Created on 17 Dec 2017

@author: jwong
'''
import QuestionProcessor
import json
import csv
from collections import Counter
from nltk import word_tokenize

class BOWQuestionProcessor(QuestionProcessor):
    def __init__(self, questionFile, vocabBOWfile):
        with open(questionFile) as qnFile:
            qnData = json.load(qnFile)

        with open(vocabBOWfile, 'rb') as bowFile:
            reader = csv.reader(bowFile, delimiter=',')
            bowDim = next(reader)

        index = 0
        bowDimMap = {}
        for word in bowDim:
            bowDimMap[word] = index
            index = index + 1 

        self.qnData = qnData
        self.bowDim = bowDim
        self.bowDimMap = bowDimMap
        self.bowLen = len(bowDim)

    def getQn(self, qnID):
        return self.qnData[str(qnID)]

    def getEncodedQn(self, qnID):
        #return bag of words vector for the qn
        qnVec = [0] * self.bowLen
        qn = self.getQn(qnID)
        for word in word_tokenize(qn.lower()):
            if (word != '?' and word in self.bowDimMap):
                qnVec[self.bowDimMap[word]] = qnVec[self.bowDimMap[word]] + 1
        return qnVec, qn
