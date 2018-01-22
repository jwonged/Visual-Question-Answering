'''
Created on 20 Jan 2018

@author: jwong
'''
import json
import csv
from nltk import word_tokenize

class InputProcessor(object):
    '''
    Generator which processes and batches input
    args:
        annotFile
        qnFile
        imgFile
        ansClassFile
        vocabFile
    '''

    def __init__(self, annotFile, qnFile, imgFile, ansClassFile, vocabFile):
        print('Reading ' + imgFile)
        self.imgData = self._readJsonFile(imgFile)
        
        print('Reading ' + annotFile)
        self.annots = self._readJsonFile(annotFile)
        
        print('Reading ' + qnFile)
        self.rawQns = self._readJsonFile(qnFile)
        
        print('Reading ' + ansClassFile)
        self.mapAnsToClass = self._loadAnsMap(ansClassFile)
        
        print('Reading ' + vocabFile)
        self.mapWordToID = self._loadVocabFromFile(vocabFile)
        
        self.index = 0
        self.epoch = 0
        
    def _readJsonFile(self, fileName):
        with open(fileName) as jsonFile:
            return json.load(jsonFile)
    
    def _loadAnsMap(self, ansClassFile):
        #loads mapping: ans --> ans class index
        with open(ansClassFile, 'rb') as ansFile:
            reader = csv.reader(ansFile, delimiter=',')
            ansVec = next(reader)
        ansClassMap = {}
        for classIndex, word in enumerate(ansVec):
            word = word.strip()
            ansClassMap[word] = classIndex
        print('Read in answer mapping with {} answers'.format(len(ansClassMap)))
        return ansClassMap
    
    def _loadVocabFromFile(self, vocabFile):
        '''
        Loads vocab file -- contains 1 word per line
        Output: dict map  word : word_id
        '''
        mapWordID = {}
        with open(vocabFile) as f:
            for word_id, word in enumerate(f):
                word = word.strip()
                mapWordID[word] = word_id
        print('Read in vocab with {} words'.format(len(mapWordID)))
        return mapWordID
    
    def _mapQnToIDs(self, qn):
        #currently ignoring words not in vocab -- need handle this
        wordList = word_tokenize(qn)
        idList = []
        for word in wordList:
            word = word.strip()
            if word in self.mapWordToID:
                idList.append(self.mapWordToID[word]) 
        return idList
    
    def getNextBatch(self, batchSize):
        batchOfQnsAsWordIDs, img_vecs, labels = [], [], []
        for annot in self.annots:
            if (len(batchOfQnsAsWordIDs) == batchSize):
                batchOfQnsAsWordIDs, qnLengths = self._padQuestionIDs(batchOfQnsAsWordIDs, 0)
                yield batchOfQnsAsWordIDs, qnLengths, img_vecs, labels
                batchOfQnsAsWordIDs, qnLengths, img_vecs, labels = [], [], [], []
                
            if self.annots[self.index]['answers'] in self.mapAnsToClass:
                annot = self.annots[self.index]
                
                #process question
                rawQn = self.rawQns[str(annot['question_id'])]
                qnAsWordIDs = self._mapQnToIDs(rawQn)
                batchOfQnsAsWordIDs.append(qnAsWordIDs)
                
                #process img
                img_id = str(annot['image_id'])
                img_vec = self.imgData[img_id][0]
                img_vecs.append(img_vec)
                
                #process label
                labelClass = self.mapAnsToClass[annot['answers']]
                labels.append(labelClass)
            
            self.index += 1
        
        if len(batchOfQnsAsWordIDs) != 0:
            batchOfQnsAsWordIDs, qnLengths = self._padQuestionIDs(batchOfQnsAsWordIDs, 0)
            yield batchOfQnsAsWordIDs, qnLengths, img_vecs, labels
    
    def getWholeBatch(self):
        batchOfQnsAsWordIDs, img_vecs, labels = [], [], []
        for annot in self.annots:
            if self.annots[self.index]['answers'] in self.mapAnsToClass:
                annot = self.annots[self.index]
                
                #process question
                rawQn = self.rawQns[str(annot['question_id'])]
                qnAsWordIDs = self._mapQnToIDs(rawQn)
                batchOfQnsAsWordIDs.append(qnAsWordIDs)
                
                #process img
                img_vec = self.imgData[str(annot['image_id'])][0]
                img_vecs.append(img_vec)
                
                #process label
                labelClass = self.mapAnsToClass[annot['answers']]
                labels.append(labelClass)
        
        if len(batchOfQnsAsWordIDs) != 0:
            batchOfQnsAsWordIDs, qnLengths = self._padQuestionIDs(batchOfQnsAsWordIDs, 0)
            return batchOfQnsAsWordIDs, qnLengths, img_vecs, labels
    
    def _padQuestionIDs(self, questions, padding):
        '''
        Pads each list to be same as max length
        args:
            questions: list of list of word IDs (ie a batch of qns)
            padding: symbol to pad with
        '''
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
    
    '''
    Deprecated nextBatch func
    def getNextBatch(self, batchSize):
        if (self.index + batchSize > self.size-1):
            batchSize = self.size - self.index - 1
            self.epoch += 1
        
        qnAsWordIDsBatch, seqLens, img_vecs, labels = [], [], [], []
         
        while (len(qnAsWordIDsBatch) < batchSize):
            if self.annots[self.index]['answers'] in self.mapAnsToClass:
                annot = self.annots[self.index]
                
                #process question
                rawQn = self.rawQns[str(annot['question_id'])]
                qnAsWordIDs = self._mapQnToIDs(rawQn)
                qnAsWordIDsBatch.append(qnAsWordIDs)
                
                #process img
                img_vec = self.imgData[str(annot['image_id'])][0]
                img_vecs.append(img_vec)
                
                #process label
                labelClass = self.mapAnsToClass[annot['answers']]
                labels.append(labelClass)
                 
            else:
                print('Not in class!!!')
            
            self.index += 1
        #pad sequence
        
        if (self.index >= self.size -1):
            self.index = 0
        return qnAsWordIDsBatch, seqLens, img_vecs, labels
    '''
    
    
        