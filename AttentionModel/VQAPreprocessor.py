'''
Created on 23 Jan 2018

@author: jwong
'''

from nltk import word_tokenize
from collections import Counter
import pickle
import json
import numpy as np

from configs import Attention_LapConfig
from utils.model_utils import AnswerProcessor

class VQAPreprocessor(object):
    '''
    1) Produce vocab for qns
    2) Shorten embeddings file
    '''
    def __init__(self, config, ansProcessor):
        self.config = config
        self.ansProcessor = ansProcessor
        
        self.trainFile = config.rawQnTrain
        self.valTestFile = config.rawQnValTestFile
        
        self.wordToIDmap = None
        self.singleCountTrainWords = None
        self.ansToClassMap = None
        self.classToAnsMap = None
    
    def preprocessVQADataset(self):
        self.preprocessAnnotationsAndSplitVal()
        self.getAnsClassMaps()
        self.getVocabForEmbeddings()
        self.saveToFile()
        self.shortenEmbeddingsFile()
    
    def _getWord2VecVocabSet(self, pretrainedw2v=None):
        if pretrainedw2v is None:
            pretrainedw2v = self.config.pretrainedw2v
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
            - all vocab in training (Non word2vec will be initialized to empty vecs)
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
        data['ansToClassMap'] = self.ansToClassMap 
        data['wordToIDmap'] = self.wordToIDmap
        data['classToAnsMap'] = self.classToAnsMap 
        
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
        annots= self._readJSONfile(fileName)
        
        listOfAllAnswers = []
        ansClasses = set()
        for annot in annots:
            for ansDic in annot['answers']:
                ans = self._processAnswer(ansDic['answer'])
                listOfAllAnswers.append(ans)
                ansClasses.add(ans)
        print('Extracted {} answers; {} unique answers from {}'.format(
            len(listOfAllAnswers), len(ansClasses), fileName))
        return listOfAllAnswers
    
    def _processAnswer(self, answer):
        answer = answer.replace('\n', ' ')
        answer = answer.replace('\t', ' ')
        answer = answer.strip()
        answer = self.ansProcessor.processPunctuation(answer)
        answer = self.ansProcessor.processDigitArticle(answer)
        return answer
    
    def _getMostFreqAnswers(self, listOfAllAnswers):
        countThreshold = 8
        n = 3000 #most freq n
        mostFreqAnswers = Counter(listOfAllAnswers).most_common(n) #tuple
        freqAnswers = set()
        numberOfCoveredAnswers = 0
        for (answer, count) in mostFreqAnswers:
            if count > countThreshold:
                freqAnswers.add(answer)
                numberOfCoveredAnswers += count
            else:
                break
        print('Most freq answers appearing more than 8 times: {}'.format(
            len(freqAnswers)))
        print('{}/{} training answers covered: {}'.format(
            numberOfCoveredAnswers,len(mostFreqAnswers), 
            100*numberOfCoveredAnswers/len(listOfAllAnswers)))
        return freqAnswers #list
    
    def getAnsClassMaps(self, trainAnnotFile=None):
        if trainAnnotFile is None:
            trainAnnotFile = self.config.trainAnnotFile
        ansClassList = self._getAnsClassesFromFile(trainAnnotFile)
        ansClasses = self._getMostFreqAnswers(ansClassList)
        
        ansToClassMap = {}
        classToAnsMap = {}
        for index, ansWord in enumerate(ansClasses):
            ansToClassMap[ansWord] = index
            classToAnsMap[index] = ansWord
        print('Extracted {} answer classes'.format(len(ansToClassMap)))
        self.ansToClassMap = ansToClassMap
        self.classToAnsMap = classToAnsMap
    
    def preprocessAnnotationsAndSplitVal(self):
        '''
        Processes annotations into a json file containing a list
        Splits val set into val and local test set
        '''
        #train set
        trainAnnots = self._readJSONfile(self.config.originalAnnotTrain)['annotations']
        with open(self.config.trainAnnotFile, 'w') as jsonOut:
            json.dump(trainAnnots, jsonOut)
        
        #val set -- split into val and test set
        valImages = self._readJSONfile(self.config.valImgFile)
        testImages = self._readJSONfile(self.config.testImgFile)
        annots = self._readJSONfile(self.config.originalAnnotVal)['annotations']
        
        valList, testList = [], []
        for annot in annots:
            if str(annot['image_id']) in valImages:
                valList.append(annot)
            elif str(annot['image_id']) in testImages:
                testList.append(annot)
            else:
                print('Error: {} not found'.format(annot['image_id']))
        
        print('Saving valtest files...')
        with open(self.config.valAnnotFile, 'w') as jsonOut:
            json.dump(valList, jsonOut)
        with open(self.config.testAnnotFile, 'w') as jsonOut:
            json.dump(testList, jsonOut)
        print('Processing complete')
        
    def _readJSONfile(self, jsonfile):
        print('Reading {}'.format(jsonfile))
        with open(jsonfile) as jsonFile:
            return json.load(jsonFile)
        
    
if __name__ == '__main__':
    ansProcessor = AnswerProcessor()
    config = Attention_LapConfig(False)
    processor = VQAPreprocessor(config, ansProcessor)
    #processor.preprocessAnnotationsAndSplitVal()
    processor.getAnsClassMaps()
    #processor.getVocabForEmbeddings()
    #processor.saveToFile()
    #processor.shortenEmbeddingsFile()
    