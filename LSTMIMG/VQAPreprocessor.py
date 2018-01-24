'''
Created on 23 Jan 2018

@author: jwong
'''
import json
from nltk import word_tokenize
from collections import Counter
import pickle
from Config import Config
import numpy as np

class VQAPreprocessor(object):
    def __init__(self, config):
        self.config = config
        
        self.trainFile = config.rawQnTrain
        self.valTestFile = config.rawQnValTestFile
        
        self.wordToIDmap = None
        self.singleCountTrainWords = None
        self.ansToClassMap = None
        self.classToAnsMap = None
        
    
    def _getWord2VecVocabSet(self):
        print('Reading {}'.format(self.config.pretrainedw2v))
        vocab = set()
        with open(self.config.pretrainedw2v) as f:
            for line in f:
                word = line.strip().split(' ')[0]
                vocab.add(word.lower())
        print('Extracted {} words from word2vec'.format(len(vocab)))
        return vocab

    def _getWordFromQnFile(self,fileName):
        #Retrieves vocab set and list of all words in qn file
        print('Reading {}'.format(fileName))
        with open(fileName) as qnFile:
            qnmap = json.load(qnFile)
        
        wordList = []
        vocab = set()
        for _,qn in qnmap.iteritems():
            for word in word_tokenize(qn):
                word = word.strip().lower()
                wordList.append(word)
                vocab.add(word)
                
        print('Extracted {} words from {}'.format(len(vocab), fileName))
        return vocab, wordList
    
    def _getSingleCountWords(self, allWords):
        #args: list of all words
        #returns set of words with count == 1
        singleCountWords = set()
        wordCounts = Counter(allWords)
        for word, count in wordCounts.iteritems():
            if count == 1:
                singleCountWords.add(word)
                
        return singleCountWords
    
    def getVocabForEmbeddings(self):
        '''
        Embedding vocab includes:
            - all vocab in training (some will be initialized to empty vecs)
            - all vocab in val/test which are in pretrained w2v
            - UNK
        '''
        trainVocab, trainWordList = self._getWordFromQnFile(self.trainFile)
        valTestVocab, _ = self._getWordFromQnFile(self.valTestFile)
        
        word2vecVocab = self._getWord2VecVocabSet()
        
        allDatasetVocab = trainVocab.union(valTestVocab)
        
        wordsForEmbeddings = allDatasetVocab.intersection(word2vecVocab)
        wordsForEmbeddings = wordsForEmbeddings.union(trainVocab)
        wordsForEmbeddings.add(self.config.unkWord)
        
        singleCountTrainWords = self._getSingleCountWords(trainWordList)
        print('All dataset vocab: {}, wordsForEmbeddingsAfterTrain: {}'.format(
            len(allDatasetVocab), len(wordsForEmbeddings)))
        print('Single count words: {}'.format(len(singleCountTrainWords)))
        
        self.wordToIDmap = {}
        for word_id, word in enumerate(wordsForEmbeddings):
            self.wordToIDmap[word] = word_id
        self.singleCountTrainWords = singleCountTrainWords
        print('map: {}'.format(len(self.wordToIDmap)))
    
    def saveToFile(self):
        data = {}
        data['singleCountWords'] = self.singleCountTrainWords
        data['ansToClassMap'] = self.ansToClassMap # not in use
        data['wordToIDmap'] = self.wordToIDmap
        data['classToAnsMap'] = self.classToAnsMap # not in use
        
        with open(self.config.preprocessedVQAMapsFile, 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        print('Saved to {}'.format(self.config.preprocessedVQAMapsFile))
        
    def shortenEmbeddingsFile(self):
        embeddingVecs = np.random.uniform(low=-1.0, 
                                          high=1.0, 
                                          size=[len(self.wordToIDmap), 300])
        
        with open(self.config.pretrainedw2v) as f:
            for line in f:
                line = line.strip().split(' ')
                word = line[0].lower()
                if word in self.wordToIDmap:
                    vec = [float(val) for val in line[1:]]
                    word_id = self.wordToIDmap[word]
                    embeddingVecs[word_id] = np.asarray(vec)
        
        np.savez_compressed(self.config.shortenedEmbeddingsWithUNKFile, vectors=embeddingVecs)
        print('Written np array of shape {} to file {}'.format(
            embeddingVecs.shape, self.config.shortenedEmbeddingsWithUNKFile))
        print(embeddingVecs[1000])

    def _getAnsClassesFromFile(self, fileName):
        print('Reading {}'.format(fileName))
        with open(fileName) as annotFile:
            annotBatch = json.load(annotFile)
        
        ansClasses = set()
        for annot in annotBatch:
            ansClasses.add(annot["answers"].lower())
        print('Extracted {} classes from {}'.format(len(ansClasses), fileName))
        return ansClasses
    
    def getAnsClassMaps(self):
        pass

if __name__ == '__main__':
    config = Config()
    processor = VQAPreprocessor(config)
    processor.getVocabForEmbeddings()
    processor.saveToFile()
    processor.shortenEmbeddingsFile()
    