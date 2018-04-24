'''
Created on 15 Jan 2018

@author: jwong
'''
import numpy as np
import re
import pickle
import csv
import json
from collections import Counter

def getPretrainedw2v(filename):
    """
    Args:
        filename: path to the npz file
    Returns:
        matrix of embeddings (np array)
    """
    with np.load(filename) as data:
        return data["vectors"]

def resolveAnswer(possibleAnswersList):
    #answers = []
    #for answerDetails in possibleAnswersList:
    #    answers.append(answerDetails['answer'])
    mostCommon = Counter(possibleAnswersList).most_common(1)
    return mostCommon[0][0]

def generateForSubmission(qn_ids, preds, jsonFile):
        '''
        result{
            "question_id": int,
            "answer": str
        }'''
        results = []
        for qn_id, pred in zip(qn_ids, preds):
            singleResult = {}
            singleResult["question_id"] = int(qn_id)
            singleResult["answer"] = str(pred)
            results.append(singleResult)
        
        with open(jsonFile, 'w') as jsonOut:
            print('Writing to {}'.format(jsonFile))
            json.dump(results, jsonOut)

def makeSmallDummyData():
    from LSTMIMG_LapConfig import LSTMIMG_LapConfig
    from LSTMIMG_GPUConfig import LSTMIMG_GPUConfig
    from TrainProcessor import LSTMIMGProcessor, TestProcessor
    config = LSTMIMG_LapConfig()
    trainReader = LSTMIMGProcessor(config.trainAnnotFile, 
                                 config.rawQnTrain, 
                                 config.trainImgFile, 
                                 config.ansClass1000File, 
                                 config,
                                 is_training=True)
    
    #dumReader = DummyReader()
    dummyData = []
    for i, (batch) in enumerate(
            trainReader.getNextBatch(32)):
        if i==100:
            break
        dummyData.append(batch)
    
    print('Completed producing dataset of size {}'.format(len(dummyData)))
    file = '/media/jwong/Transcend/VQADataset/DummySets/dummyTupleBatchesLSTMIMG.pkl'
    with open(file, 'wb') as f:
            pickle.dump(dummyData, f, protocol=pickle.HIGHEST_PROTOCOL)
    print('Printed to file')
    


class DummyReader():
    def __init__(self, config):
        file = '/media/jwong/Transcend/VQADataset/DummySets/dummyTupleBatchesLSTMIMG.pkl'
        with open(file, 'rb') as jFile:
            print('Reading {}'.format(file))
            self.tupList = pickle.load(jFile)[:30]
            
            
        print('Reading ' + config.ansClass1000File)
        self.mapAnsToClass, self.classToAnsMap = self._loadAnsMap(config.ansClass1000File)
    
    def getNextBatch(self, batch_size):
        for tup in self.tupList:
            yield tup
    
    def _loadAnsMap(self, ansClassFile):
        #loads mapping: ans --> ans class index
        with open(ansClassFile, 'rb') as ansFile:
            reader = csv.reader(ansFile, delimiter=',')
            ansVec = next(reader)
        classToAnsMap = {}
        ansClassMap = {}
        for classIndex, word in enumerate(ansVec):
            word = word.strip()
            ansClassMap[word] = classIndex
            classToAnsMap[classIndex] = word
        print('Read in answer mapping with {} answers'.format(len(ansClassMap)))
        return ansClassMap, classToAnsMap
    
    def getAnsMap(self):
        return self.classToAnsMap
    
class AnswerProcessor:
    '''
    Normalises answers towards a canonical form:
    Adapted and following specifications from VQA's official evaluation code:
    http://visualqa.org/evaluation.html
    '''
    def __init__(self):
        self.contractions = {"aint": "ain't", "arent": "aren't", "cant": "can't", "couldve": "could've", "couldnt": "couldn't", \
                             "couldn'tve": "couldn't've", "couldnt've": "couldn't've", "didnt": "didn't", "doesnt": "doesn't", "dont": "don't", "hadnt": "hadn't", \
                             "hadnt've": "hadn't've", "hadn'tve": "hadn't've", "hasnt": "hasn't", "havent": "haven't", "hed": "he'd", "hed've": "he'd've", \
                             "he'dve": "he'd've", "hes": "he's", "howd": "how'd", "howll": "how'll", "hows": "how's", "Id've": "I'd've", "I'dve": "I'd've", \
                             "Im": "I'm", "Ive": "I've", "isnt": "isn't", "itd": "it'd", "itd've": "it'd've", "it'dve": "it'd've", "itll": "it'll", "let's": "let's", \
                             "maam": "ma'am", "mightnt": "mightn't", "mightnt've": "mightn't've", "mightn'tve": "mightn't've", "mightve": "might've", \
                             "mustnt": "mustn't", "mustve": "must've", "neednt": "needn't", "notve": "not've", "oclock": "o'clock", "oughtnt": "oughtn't", \
                             "ow's'at": "'ow's'at", "'ows'at": "'ow's'at", "'ow'sat": "'ow's'at", "shant": "shan't", "shed've": "she'd've", "she'dve": "she'd've", \
                             "she's": "she's", "shouldve": "should've", "shouldnt": "shouldn't", "shouldnt've": "shouldn't've", "shouldn'tve": "shouldn't've", \
                             "somebody'd": "somebodyd", "somebodyd've": "somebody'd've", "somebody'dve": "somebody'd've", "somebodyll": "somebody'll", \
                             "somebodys": "somebody's", "someoned": "someone'd", "someoned've": "someone'd've", "someone'dve": "someone'd've", \
                             "someonell": "someone'll", "someones": "someone's", "somethingd": "something'd", "somethingd've": "something'd've", \
                             "something'dve": "something'd've", "somethingll": "something'll", "thats": "that's", "thered": "there'd", "thered've": "there'd've", \
                             "there'dve": "there'd've", "therere": "there're", "theres": "there's", "theyd": "they'd", "theyd've": "they'd've", \
                             "they'dve": "they'd've", "theyll": "they'll", "theyre": "they're", "theyve": "they've", "twas": "'twas", "wasnt": "wasn't", \
                             "wed've": "we'd've", "we'dve": "we'd've", "weve": "we've", "werent": "weren't", "whatll": "what'll", "whatre": "what're", \
                             "whats": "what's", "whatve": "what've", "whens": "when's", "whered": "where'd", "wheres": "where's", "whereve": "where've", \
                             "whod": "who'd", "whod've": "who'd've", "who'dve": "who'd've", "wholl": "who'll", "whos": "who's", "whove": "who've", "whyll": "why'll", \
                             "whyre": "why're", "whys": "why's", "wont": "won't", "wouldve": "would've", "wouldnt": "wouldn't", "wouldnt've": "wouldn't've", \
                             "wouldn'tve": "wouldn't've", "yall": "y'all", "yall'll": "y'all'll", "y'allll": "y'all'll", "yall'd've": "y'all'd've", \
                             "y'alld've": "y'all'd've", "y'all'dve": "y'all'd've", "youd": "you'd", "youd've": "you'd've", "you'dve": "you'd've", \
                             "youll": "you'll", "youre": "you're", "youve": "you've"}
        self.manualMap    = { 'none': '0',
                              'zero': '0',
                              'one': '1',
                              'two': '2',
                              'three': '3',
                              'four': '4',
                              'five': '5',
                              'six': '6',
                              'seven': '7',
                              'eight': '8',
                              'nine': '9',
                              'ten': '10'
                            }
        self.articles     = ['a',
                             'an',
                             'the'
                            ]
 

        self.periodStrip  = re.compile("(?!<=\d)(\.)(?!\d)")
        self.commaStrip   = re.compile("(\d)(\,)(\d)")
        self.punct        = [';', r"/", '[', ']', '"', '{', '}',
                             '(', ')', '=', '+', '\\', '_', '-',
                             '>', '<', '@', '`', ',', '?', '!']
    
    def processAnswer(self, answer):
        answer = answer.replace('\n', ' ')
        answer = answer.replace('\t', ' ')
        answer = answer.strip()
        answer = self._processPunctuation(answer)
        answer = self._processDigitArticle(answer)
        return answer
    
    def _processPunctuation(self, inText):
        outText = inText
        for p in self.punct:
            if (p + ' ' in inText or ' ' + p in inText) or (re.search(self.commaStrip, inText) != None):
                outText = outText.replace(p, '')
            else:
                outText = outText.replace(p, ' ')    
        outText = self.periodStrip.sub("",
                                      outText,
                                      re.UNICODE)
        return outText
    
    def _processDigitArticle(self, inText):
        outText = []
        tempText = inText.lower().split()
        for word in tempText:
            word = self.manualMap.setdefault(word, word)
            if word not in self.articles:
                outText.append(word)
            else:
                pass
        for wordId, word in enumerate(outText):
            if word in self.contractions: 
                outText[wordId] = self.contractions[word]
        outText = ' '.join(outText)
        return outText