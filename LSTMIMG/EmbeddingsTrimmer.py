'''
Created on 14 Jan 2018

@author: jwong
'''

import json
from collections import Counter
import numpy as np
from nltk import word_tokenize

def _resolveAnswer(possibleAnswersList):
        answers = []
        for answerDetails in possibleAnswersList:
            answers.append(answerDetails['answer'])
        mostCommon = Counter(answers).most_common(1)
        return mostCommon[0][0]

def _resolveAnnots(inputFile, outputFile):
    print('Reading {}'.format(inputFile))
    with open(inputFile) as annotFile:
        annotBatch = json.load(annotFile)
    
    resolvedBatch = []
    i = 0
    for annot in annotBatch:
        singleAns = _resolveAnswer(annot['answers'])
        annot['answers'] = singleAns
        resolvedBatch.append(annot)
        i += 1
    
    print('Resolved {} answers'.format(i))
    
    with open(outputFile, 'w') as jsonOut:
        json.dump(resolvedBatch, jsonOut)
    print('Written to file {}'.format(outputFile))

def resolveAnnots():
    '''
    Input: json file in type AllTrainAnnotationsList.json
    Simplifies json file to following format with only a single resolved answer
    Output:
    [
        {
            "question_id" : int,
            "image_id" : int,
            "question_type" : str,
            "answer_type" : str,
            "answers" : answer(str),
            "multiple_choice_answer" : str
        }, ...
    ]
    '''
    annotationsTrain = '/media/jwong/Transcend/VQADataset/TrainSet/AllTrainAnnotationsList.json'
    trainAnnotOut = '/media/jwong/Transcend/VQADataset/TrainSet/LSTMIMGData/AllTrainAnnotResolvedList.json'
    _resolveAnnots(annotationsTrain, trainAnnotOut)
    
    annotationsVal = '/media/jwong/Transcend/VQADataset/ValTestSet/AllValAnnotationsList.json'
    valAnnotOut = '/media/jwong/Transcend/VQADataset/ValTestSet/LSTMIMGData/AllValAnnotResolvedList.json'
    _resolveAnnots(annotationsVal, valAnnotOut)
    
    annotationsTest = '/media/jwong/Transcend/VQADataset/ValTestSet/AllTestAnnotationsList.json'
    testAnnotOut = '/media/jwong/Transcend/VQADataset/ValTestSet/LSTMIMGData/AllTestAnnotResolvedList.json'
    _resolveAnnots(annotationsTest, testAnnotOut)
    

def _getWord2VecVocabSet():
    pretrainedw2v = '/media/jwong/Transcend/GoogleNews-vectors-negative300.txt'
    print('Reading {}'.format(pretrainedw2v))
    vocab = set()
    with open(pretrainedw2v) as f:
        for line in f:
            word = line.strip().split(' ')[0]
            vocab.add(word)
    print('Extracted {} words from word2vec'.format(len(vocab)))
    return vocab

def _getVocabFromQnFile(fileName):
    print('Reading {}'.format(fileName))
    with open(fileName) as qnFile:
        qnmap = json.load(qnFile)
    
    vocab = set()
    for _,qn in qnmap.iteritems():
        for word in word_tokenize(qn):
            vocab.add(word)
    print('Extracted {} words from {}'.format(len(vocab), fileName))
    return vocab
    
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

def getVocabFromFile():
    '''
    Loads vocab file -- contains 1 word per line
    Output: dict map  word : word_id
    '''
    vocabFile = '/media/jwong/Transcend/VQADataset/FullVQAVocab.txt'
    mapWordID = {}
    with open(vocabFile) as f:
        for word_id, word in enumerate(f):
            word = word.strip()
            mapWordID[word] = word_id
    print('Read in vocab with {} words'.format(len(mapWordID)))
    return mapWordID
    
def shortenEmbeddings():
    '''
    Load full vocab, and retrieve word2vec vectors corresponding to vocab
    Produce shortened version of embeddings file
    Contains vocab word embeddings, with word_id correspondingn to vect index
    '''
    pretrainedw2v = '/media/jwong/Transcend/GoogleNews-vectors-negative300.txt'
    shortenedEmbeddingsFile = '/media/jwong/Transcend/VQADataset/cutW2VEmbeddings.npz'
    
    vocabMap = getVocabFromFile()
    vectors = np.zeros([len(vocabMap), 300]) #word2vec 300-D vectors
    
    with open(pretrainedw2v) as f:
        for line in f:
            #retrieve embedding vector
            line = line.strip().split(' ')
            vect = [float(val) for val in line[1:]]
            word = line[0]
            if word in vocabMap:
                word_id = vocabMap[word]
                vectors[word_id] = np.asarray(vect)
                
    np.savez_compressed(shortenedEmbeddingsFile, vectors=vectors)
    print('Written np array of shape {} to file {}'.format(vectors.shape, shortenedEmbeddingsFile))

def _processQnStrToID(qn, mapWordID):
    ids = []
    for word in word_tokenize(qn):
        if word in mapWordID:
            ids.append(mapWordID[word])
    return ids

def _convertQnToWordIDs(fileName, outFile):
    print('Reading {}'.format(fileName))
    mapWordID = getVocabFromFile()
    with open(fileName) as qnFile:
        qnmap = json.load(qnFile)
    
    resultMap = {}
    for qn_id,qn in qnmap.iteritems():
        resultMap[qn_id] = _processQnStrToID(qn, mapWordID)
        
    print('Writing to {}'.format(fileName))
    with open(outFile, 'w') as jsonOut:
        json.dump(resultMap, jsonOut)
    
    count = 0
    for qn_id,qn in resultMap.iteritems():
        print('{} : {}'.format(qn_id, qn))
        count += 1
        if count > 5:
            break

def convertQnFileToWordIDs():
    '''
    Process qn file of mapping to raw qns to a mapping to list of word ids
    Input: Qn_ID --> raw str qn
    Output: Qn_ID --> list of word_ids
   '''
    strQnTrain = '/media/jwong/Transcend/VQADataset/TrainSet/Questions_Train_mscoco/Preprocessed/processedOpenEnded_trainQns.json'
    strValTestQnFile = '/media/jwong/Transcend/VQADataset/ValTestSet/Questions_Val_mscoco/preprocessedValTestQnsOpenEnded.json'
    
    idQnTrain = '/media/jwong/Transcend/VQADataset/TrainSet/Questions_Train_mscoco/Preprocessed/QnWordsIDlistTrain_OpenEnded.json'
    idValTestQnFile = '/media/jwong/Transcend/VQADataset/ValTestSet/Questions_Val_mscoco/QnWordsIDlistValTest_OpenEnded.json'
    
    _convertQnToWordIDs(strQnTrain, idQnTrain)
    _convertQnToWordIDs(strValTestQnFile,idValTestQnFile)
    print('Done str to id conversion')
    
def preprocessDataset():
    #simplify annotations json and resolve 10 answers to 1
    resolveAnnots()
    
    #Get full vocab from all qn and ans files, and produce intersection with word2vec
    getAllDatasetVocab()
    
    #use vocab to trim embeddings file
    shortenEmbeddings()
    
    #Turn raw str qns into list of word_ids
    convertQnFileToWordIDs()
    
if __name__ == '__main__':
    pass
    
    