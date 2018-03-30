'''
Created on 30 Mar 2018

@author: jwong
'''
import json
class InputProcessor(object):
    """"
    Base Input Processor
    """
    def __init__(self,config):
        self.config = config
        
    def _readJsonFile(self, fileName):
        print('Reading {}'.format(fileName))
        with open(fileName) as jsonFile:
            return json.load(jsonFile)
    
    def _padQuestionIDs(self, questions, padding):
        """
        Pads each list to be same as max length
        args:
            questions: list of list of word IDs (ie a batch of qns)
            padding: symbol to pad with
        """
        maxLength = max(map(lambda x : len(x), questions))
        #Get length of longest qn
        paddedQuestions, qnLengths = [], []
        for qn in questions:
            qn = list(qn) #ensure list format
            if (len(qn) < maxLength):
                paddedQn = qn + [padding]*(maxLength - len(qn))
                paddedQuestions.append(paddedQn)
            else:
                paddedQuestions.append(qn)
            qnLengths.append(len(qn))
            
        return paddedQuestions, qnLengths
    
    def destruct(self):
        pass