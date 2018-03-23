'''
Created on 15 Jan 2018

@author: jwong
'''
import numpy as np
import re
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

import matplotlib.pyplot as plt
import skimage.transform
import cv2
from scipy import ndimage
import numpy as np
from textwrap import wrap
class OutputGenerator(object):
    def __init__(self, imgPathsFile):
        self.idToImgpathMap = {}
        print('Reading {}'.format(imgPathsFile))
        with open(imgPathsFile, 'r') as reader:
            for image_path in reader:
                image_path = image_path.strip()
                self.idToImgpathMap[str(self.getImageID(image_path))] = image_path
    
    def getImageID(self,image_path):
        #Get image ID
        splitPath = image_path.split('/')
        imgNameParts = splitPath[len(splitPath)-1].split('_') #COCO, train, XXX.jpg
        suffix =  imgNameParts[len(imgNameParts)-1] #XXX.jpg
        img_id = int(suffix.split('.')[0])
        return img_id
    
    def convertIDtoPath(self, img_id):
        return self.idToImgpathMap[img_id]
    
    def displaySingleOutput(self, alpha, img_id, qn, pred):
        print('Num of images: {}'.format(img_id))
        imgvec = ndimage.imread(self.idToImgpathMap[img_id])
        imgvec = cv2.resize(imgvec, dsize=(448,448))
        
        alp_img = skimage.transform.pyramid_expand(
            alpha.reshape(14,14), upscale=32, sigma=20)
        alp_img = np.transpose(alp_img, (1,0))
        
        plt.subplot(1,1,1)
        plt.title('Qn: {}, pred: {}'.format(qn, pred))
        plt.imshow(imgvec)
        plt.imshow(alp_img, alpha=0.80)
        plt.axis('off')
        
        #plt.subplot(2,1,1)
        #plt.title('Qn: {}, pred: {}'.format(qn, pred))
        #plt.imshow(imgvec)
        #plt.axis('off')
            
        plt.show()
        
    def displayOutput(self, alphas, img_ids, qns, preds):
        
        
        print('Num of images: {}'.format(img_ids))
        fig = plt.figure()
        for n, (alp, img_id, qn, pred) in enumerate(zip(alphas, img_ids, qns, preds)):
            if n>2:
                break
            imgvec = ndimage.imread(self.idToImgpathMap[img_id])
            imgvec = cv2.resize(imgvec, dsize=(448,448))
            
            alp_img = skimage.transform.pyramid_expand(
                alp.reshape(14,14), upscale=32, sigma=20)
            alp_img = np.transpose(alp_img, (1,0))
            
            
            plt.subplot(2,4,(n+1))
            plt.title("\n".join(wrap(
                "Qn: {} Pred: {}".format(qn, pred), 20)))
            plt.imshow(imgvec)
            plt.imshow(alp_img, alpha=0.80)
            plt.axis('off')
            
            plt.subplot(2,4,(n+1)*2)
            plt.title('Qn: {}, pred: {}'.format(qn, pred))
            plt.imshow(imgvec)
            plt.axis('off')
        #plt.tight_layout()
        plt.show()

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