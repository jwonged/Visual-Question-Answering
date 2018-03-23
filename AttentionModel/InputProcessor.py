'''
Created on 20 Jan 2018

@author: jwong
'''
import json
import csv
from nltk import word_tokenize
import pickle
import numpy as np
import random
import shelve

class InputProcessor(object):
    '''
    For attention model
    Generator which processes and batches input
    args:
        annotFile
        qnFile
        imgFile
        ansClassFile
        vocabFile
    '''
    
    def __init__(self, annotFile, qnFile, imgFile, config, is_training):
        print('Reading {}'.format(imgFile))
        self.imgData = shelve.open(imgFile, flag='r', protocol=pickle.HIGHEST_PROTOCOL)
        self.annots = self._readJsonFile(annotFile)
        self.rawQns = self._readJsonFile(qnFile)
        
        self.mapAnsToClass = config.mapAnsToClass
        self.classToAnsMap = config.classToAnsMap
        self.classToAnsMap[-1] = -1
        self.mapWordToID = config.mapWordToID
        self.singleCountWords = config.singleCountWords
        
        self.is_training = is_training
        self.config = config
        if config.shuffle and is_training:
            random.shuffle(self.annots)
        self.datasetSize = len(self.annots)
        
    def _readJsonFile(self, fileName):
        print('Reading {}'.format(fileName))
        with open(fileName) as jsonFile:
            return json.load(jsonFile)
    
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
                img_vec = self.imgData[img_id]
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
    
    def destruct(self):
        self.imgData.close()

class TestProcessor(object):
    '''
    For reading and processing Test dataset (without annotations)
    For submission to test server
    args:
    '''

    def __init__(self, qnFile, imgFile, config):
        print('Reading {}'.format(imgFile))
        self.imgData = shelve.open(imgFile, flag='r', protocol=pickle.HIGHEST_PROTOCOL)
        
        print('Reading' + qnFile)
        with open(qnFile) as jsonFile:
            self.qnData = json.load(jsonFile)['questions']
        
        self.config  = config
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
            img_vec = self.imgData[img_id]
            img_vecs.append(img_vec)
            img_ids.append(img_id)
            
        if len(batchOfQnsAsWordIDs) != 0:
            batchOfQnsAsWordIDs, qnLengths = self._padQuestionIDs(batchOfQnsAsWordIDs, 0)
            yield batchOfQnsAsWordIDs, qnLengths, img_vecs, rawQns, img_ids, qn_ids
            
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
    
    def destruct(self):
        self.imgData.close()
        
class OnlineProcessor(object):
    
    def __init__(self, imgFile, config):
        print('Reading {}'.format(imgFile))
        self.imgData = shelve.open(imgFile, flag='r', protocol=pickle.HIGHEST_PROTOCOL)
        
        self.config  = config
        self.classToAnsMap = config.classToAnsMap
        self.classToAnsMap[-1] = -1
        self.mapWordToID = config.mapWordToID
    
    def processInput(self, qn, img_id):
        qnAsWordIDs = self._mapQnToIDs(qn)
        img_vec = self.imgData[img_id]
        batchOfQnsAsWordIDs, qnLengths = self._padQuestionIDs([qnAsWordIDs], 0)
        return batchOfQnsAsWordIDs, qnLengths, [img_vec]
        
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
            img_vec = self.imgData[img_id]
            img_vecs.append(img_vec)
            img_ids.append(img_id)
            
        if len(batchOfQnsAsWordIDs) != 0:
            batchOfQnsAsWordIDs, qnLengths = self._padQuestionIDs(batchOfQnsAsWordIDs, 0)
            yield batchOfQnsAsWordIDs, qnLengths, img_vecs, rawQns, img_ids, qn_ids
            
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
    
    def destruct(self):
        self.imgData.close()

import caffe
import sys, os
class ImageProcessor(object):
    '''
    Processing images for E2E architecture
    '''

    def __init__(self):
        caffe_root = '/home/joshua/caffe/'
        #caffe_root = '/home/jwong/caffe/'
         
        # Model prototxt file
        self.model_prototxt = caffe_root + 'models/211839e770f7b538e2d8/VGG_ILSVRC_19_layers_deploy.prototxt'
         
        # Model caffemodel file
        self.model_trained = caffe_root + 'models/211839e770f7b538e2d8/VGG_ILSVRC_19_layers.caffemodel'
         
        # File containing the class labels
        self.imagenet_labels = caffe_root + 'data/ilsvrc12/synset_words.txt'
         
        # Path to the mean image (used for input processing)
        self.mean_path = caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy'
         
        # Name of the layer we want to extract
        self.layer_name = 'conv5_3'
        #layer_name = 'fc7'
         
        sys.path.insert(0, caffe_root + 'python')
        
        caffe.set_device(0)
        #caffe.set_mode_gpu()
        caffe.set_mode_cpu()
        
        self.net = caffe.Classifier(self.model_prototxt, self.model_trained,
                           mean=np.load(self.mean_path).mean(1).mean(1),
                           channel_swap=(2,1,0),
                           raw_scale=255,
                           image_dims=(448, 448))
        with open(self.imagenet_labels) as f:
            self.labels = f.readlines()
    
    def _getImageID(self, image_path):
        #Get image ID
        splitPath = image_path.split('/')
        imgNameParts = splitPath[len(splitPath)-1].split('_') #COCO, train, XXX.jpg
        suffix =  imgNameParts[len(imgNameParts)-1] #XXX.jpg
        img_id = int(suffix.split('.')[0])
        return img_id
    
    def processSingleImage(self, imageFile, getID=False):
        print('Extracting from layer: {}'.format(self.layer_name))
        input_image = caffe.io.load_image(imageFile.strip())
        prediction = self.net.predict([input_image], oversample=False)
        msg = ('{} : {} ( {} )'.format(imageFile.split('/')[-1], 
                                       self.labels[prediction[0].argmax()].strip(), 
                                       prediction[0][prediction[0].argmax()]))
        featureData = self.net.blobs[self.layer_name].data[0]
        
        if getID:
            img_id = self._getImageID(imageFile)
        else:
            img_id = -1
            
        return featureData, img_id
        