'''
Created on 16 Dec 2017

@author: jwong
'''

import json
import csv
from nltk import word_tokenize
import pickle
import numpy as np
import random
import shelve
from Input_Processor import InputProcessor

class BOWIMGProcessor(InputProcessor):
    '''
    For BOWIMG model
    '''
    def __init__(self, annotFile, qnFile, imgFile, config, is_training):
        super(BOWIMGProcessor, self).__init__(config)
        
        self.imgData = self._readJsonFile(imgFile)
        self.annots = self._readJsonFile(annotFile)
        self.rawQns = self._readJsonFile(qnFile)
        
        self.mapAnsToClass = config.mapAnsToClass
        self.classToAnsMap = config.classToAnsMap
        self.classToAnsMap[-1] = -1
        self.mapWordToID = config.mapWordToID
        self.singleCountWords = config.singleCountWords
        
        self.is_training = is_training
        if config.shuffle and is_training:
            random.shuffle(self.annots)
        self.datasetSize = len(self.annots)
        
    
    def _mapQnToIDs(self, qn):
        #Convert str question to a list of word_ids
        idList = []
        for word in word_tokenize(qn):
            word = word.strip().lower()
            if word in self.mapWordToID:
                #prob chance of converting a single count word to UNK
                if (self.is_training and word in self.singleCountWords and 
                        np.random.uniform() < self.config.probSingleToUNK):
                    idList.append(self.mapWordToID[self.config.unkWord])
                else:
                    idList.append(self.mapWordToID[word]) 
            else:
                if self.is_training:
                    print('This should never be printed - all train words in map')
                idList.append(self.mapWordToID[self.config.unkWord])
        return idList
    
    def getNextBatch(self, batchSize):
        batchOfQnsAsWordIDs, img_vecs, labels, rawQns, img_ids, qn_ids = [], [], [], [], [], []
        for annot in self.annots:
            if (len(batchOfQnsAsWordIDs) == batchSize):
                batchOfQnsAsWordIDs, qnLengths = self._padQuestionIDs(batchOfQnsAsWordIDs, 0)
                yield batchOfQnsAsWordIDs, qnLengths, img_vecs, labels, rawQns, img_ids, qn_ids
                batchOfQnsAsWordIDs, qnLengths, img_vecs, labels, rawQns, img_ids, qn_ids = [], [], [], [], [], [], []
            
            #Leave out answers not in AnsClass for training; map to special num for val
            if (not self.is_training) or (
                self.is_training and annot['answers'] in self.mapAnsToClass):
                #process question
                rawQn = self.rawQns[str(annot['question_id'])]
                qnAsWordIDs = self._mapQnToIDs(rawQn)
                batchOfQnsAsWordIDs.append(qnAsWordIDs)
                rawQns.append(rawQn)
                qn_ids.append(annot['question_id'])
                
                #process img
                img_id = str(annot['image_id'])
                img_vec = self.imgData[img_id][0]
                img_vecs.append(img_vec)
                img_ids.append(img_id)
                
                #process label
                if annot['answers'] not in self.mapAnsToClass:
                    if self.is_training:
                        raise ValueError('Inconsistent State in processing label')
                    labelClass = -1
                else:
                    labelClass = self.mapAnsToClass[annot['answers']]
                labels.append(labelClass)
        
        if self.config.shuffle and self.is_training:
            random.shuffle(self.annots)
        if len(batchOfQnsAsWordIDs) != 0:
            batchOfQnsAsWordIDs, qnLengths = self._padQuestionIDs(batchOfQnsAsWordIDs, 0)
            yield batchOfQnsAsWordIDs, qnLengths, img_vecs, labels, rawQns, img_ids, qn_ids
    
    def destruct(self):
        pass

class TestProcessor(InputProcessor):
    '''
    For reading and processing Test dataset (without annotations)
    For submission to test server
    args:
    '''
    
    def __init__(self, qnFile, imgFile, config):
        super(TestProcessor, self).__init__(config)
        
        self.imgData = self._readJsonFile(imgFile)
        self.qnData = self._readJsonFile(qnFile)['questions']
        
        self.classToAnsMap = config.classToAnsMap
        self.classToAnsMap[-1] = -1
        self.mapWordToID = config.mapWordToID

    def getNextBatch(self, batchSize):
        batchOfQnsAsWordIDs, img_vecs, rawQns, img_ids, qn_ids = [], [], [], [], []
        for qn in self.qnData:
            if (len(batchOfQnsAsWordIDs) == batchSize):
                batchOfQnsAsWordIDs, qnLengths = self._padQuestionIDs(batchOfQnsAsWordIDs, 0)
                yield batchOfQnsAsWordIDs, qnLengths, img_vecs, rawQns, img_ids, qn_ids
                batchOfQnsAsWordIDs, qnLengths, img_vecs, rawQns, img_ids, qn_ids = [], [], [], [], [], []
            
            #process question
            qn_id = str(qn['question_id'])
            qnStr = qn['question']
            qnAsWordIDs = self._mapQnToIDs(qnStr)
            batchOfQnsAsWordIDs.append(qnAsWordIDs)
            rawQns.append(qnStr)
            qn_ids.append(qn_id)
            
            #process img
            img_id = str(qn['image_id'])
            img_vec = self.imgData[img_id][0]
            img_vecs.append(img_vec)
            img_ids.append(img_id)
            
        if len(batchOfQnsAsWordIDs) != 0:
            batchOfQnsAsWordIDs, qnLengths = self._padQuestionIDs(batchOfQnsAsWordIDs, 0)
            yield batchOfQnsAsWordIDs, qnLengths, img_vecs, rawQns, img_ids, qn_ids
            
        
    def _mapQnToIDs(self, qn):
        #Convert str question to a list of word_ids
        idList = []
        for word in word_tokenize(qn):
            word = word.strip().lower()
            if word in self.mapWordToID:
                    idList.append(self.mapWordToID[word]) 
            else:
                idList.append(self.mapWordToID[self.config.unkWord])
        return idList
    
        
    