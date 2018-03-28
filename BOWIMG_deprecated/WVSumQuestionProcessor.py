'''
Created on 17 Dec 2017

@author: jwong

Word2Vec version of QuestionProcessor
Naive summing of feature vectors for each word in question
'''

import json

class WVSumQuestionProcessor():
    def __init__(self, questionFile):
        with open(questionFile) as qnFile:
            qnFeatures = json.load(qnFile)
        self.qnFeatures = qnFeatures

    #return feature vector corresponding to qnID
    def getEncodedQn(self, qnID):
        return self.qnFeatures[str(qnID)]
        