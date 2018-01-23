'''
Created on 23 Jan 2018

@author: jwong
'''
import json
from nltk import word_tokenize
from collections import Counter

class VQAPreprocessor(object):
    '''
    classdocs
    '''


    def __init__(self, config):
        self.config = config
    
    def _getWord2VecVocabSet(self):
        print('Reading {}'.format(self.config.pretrainedw2v))
        vocab = set()
        with open(self.config.pretrainedw2v) as f:
            for line in f:
                word = line.strip().split(' ')[0]
                vocab.add(word)
        print('Extracted {} words from word2vec'.format(len(vocab)))
        return vocab

    def _getWordFreqsFromQnFile(self,fileName):
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
        singleCountWords = set()
        wordCounts = Counter(allWords)
        numWords = 0
        for word, count in wordCounts.iteritems():
            numWords +=1
            if count == 1:
                singleCountWords.add(word)
                
        return singleCountWords
    
    def _getVocabFromAnsFile(fileName):
        print('Reading {}'.format(fileName))
        with open(fileName) as annotFile:
            annotBatch = json.load(annotFile)
        
        vocab = set()
        for annot in annotBatch:
            for word in word_tokenize(annot["answers"]):
                vocab.add(word)
        print('Extracted {} words from {}'.format(len(vocab), fileName))
        return vocab
        
    def getAllDatasetVocab():
        
        vocab.add('<UNK>')
        """Writes a vocab to a file
        Writes one word per line.
        Args:
            vocab: iterable that yields word
            filename: path to vocab file
        Returns:
            write a word per line
        """
        vocabOut = '/media/jwong/Transcend/VQADataset/FullVQAVocab.txt'
        datasetVocab = set()
        
        #add question vocab to set
        qnTrain = '/media/jwong/Transcend/VQADataset/TrainSet/Questions_Train_mscoco/Preprocessed/processedOpenEnded_trainQns.json'
        valTestQnFile = '/media/jwong/Transcend/VQADataset/ValTestSet/Questions_Val_mscoco/preprocessedValTestQnsOpenEnded.json'
        datasetVocab = datasetVocab.union(_getVocabFromQnFile(qnTrain))
        datasetVocab = datasetVocab.union(_getVocabFromQnFile(valTestQnFile))
        print('Set now contains {} words'.format(len(datasetVocab)))
        
        #add answer vocab to set
        trainAnnotOut = '/media/jwong/Transcend/VQADataset/TrainSet/LSTMIMGData/AllTrainAnnotResolvedList.json'
        valAnnotOut = '/media/jwong/Transcend/VQADataset/ValTestSet/LSTMIMGData/AllValAnnotResolvedList.json'
        testAnnotOut = '/media/jwong/Transcend/VQADataset/ValTestSet/LSTMIMGData/AllTestAnnotResolvedList.json'
        datasetVocab = datasetVocab.union(_getVocabFromAnsFile(trainAnnotOut))
        datasetVocab = datasetVocab.union(_getVocabFromAnsFile(valAnnotOut))
        datasetVocab = datasetVocab.union(_getVocabFromAnsFile(testAnnotOut))
        print('Set now contains {} words'.format(len(datasetVocab)))
        
        #get intersection of word2vec vocab and dataset vocab
        datasetVocab = datasetVocab.intersection(_getWord2VecVocabSet())
        
        print("Writing {} words to {}".format(len(datasetVocab), vocabOut))
        with open(vocabOut, "w") as f:
            for index, word in enumerate(datasetVocab, 1):
                if index != len(datasetVocab):
                    f.write("{}\n".format(word))
                else:
                    f.write(word)
                    print('Written {} words'.format(index))
        print("Completed writing {} words".format(len(datasetVocab)))