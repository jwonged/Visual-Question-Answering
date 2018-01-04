'''
Created on 17 Dec 2017

@author: jwong

Preprocess raw questions into word2vec feature vectors
Produces JSON mapping
{Qn_id : feature_vec
} 
'''
import gensim
import json
import numpy as np
from nltk import word_tokenize

pretrainedw2v = '/media/jwong/Transcend/GoogleNews-vectors-negative300.bin'
model = gensim.models.KeyedVectors.load_word2vec_format(pretrainedw2v, binary=True)

skippedWords = set()
#tokenize and convert to feature vector
def extract(question):
    vec = np.zeros(300)
    for word in word_tokenize(question):
        if (word != '?' and word in model.wv):
            current = model.wv[word]
            vec += current
        else:
            skippedWords.add(word)
    return vec.tolist()
    
def processFile(fileName, outFile):
    with open(fileName) as qnFile:
        qnmap = json.load(qnFile)
    
    count = 0
    resultMap = {}
    for key,value in qnmap.iteritems():
        resultMap[key] = extract(value)
        count += 1
        if (count%50000 == 0):
            print('Processed: {}'.format(count))
    print('Completed file {}, \n processed: {}'.format(fileName, count))
    
    with open(outFile, 'w') as jsonOut:
        json.dump(resultMap, jsonOut)
    

if __name__ == '__main__':
    print('loading train file...')
    trainQnsFile = '/media/jwong/Transcend/VQADataset/TrainSet/Questions_Train_mscoco/Preprocessed/processedOpenEnded_trainQns.json'
    trainOut = '/media/jwong/Transcend/VQADataset/TrainSet/ExtractedQnFeatures/word2VecAddQnFeatures_Train.json'
    processFile(trainQnsFile, trainOut)
    print('loading valtest file...')
    valTestQnFile = '/media/jwong/Transcend/VQADataset/ValTestSet/Questions_Val_mscoco/preprocessedValTestQnsOpenEnded.json'
    valTestOut = '/media/jwong/Transcend/VQADataset/ValTestSet/ExtractedQnFeatures/word2VecAddQnFeatures_valTest.json'
    processFile(valTestQnFile, valTestOut)
    
    with open('skippedWordsLog.txt', 'w') as f:
        f.write('\n'.join(skippedWords))
    print('Completed')