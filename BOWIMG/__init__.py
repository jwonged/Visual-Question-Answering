import json
import csv
import numpy as np
from collections import Counter
from nltk import word_tokenize
import QuestionProcessor.py
'''
Read in annotations
Retrieve answer
'''

mostFreqAnswersFile = '/home/jwong/Documents/LinuxWorkspace/Visual-Question-Answering/resources/1000MostFreqAnswers.csv'
vocabBOWfile = '/home/jwong/Documents/LinuxWorkspace/Visual-Question-Answering/resources/BOWdimensions.csv'

class InputProcessor:
	def __init__(self, questionFile, vocabBOWfile):
		self.qnProcessor = QuestionProcessor(questionFile, vocabBOWfile)

	def readAnnotationsFile(self):
		ansClasses, ansClassMap = self.get1000MostFreqAnswers()
		ansClassLen = len(ansClasses)

		#Get answer
		ans1 = self.resolveAnswer(self.annotData['annotations'][0]['answers'])
		ans2 = self.resolveAnswer(self.annotData['annotations'][1]['answers'])
		ans1e = self.encodeAns(ans1, ansClassMap, ansClassLen)
		ans2e = self.encodeAns(ans2, ansClassMap, ansClassLen)

		#Get qn
		qn1 = str(self.annotData['annotations'][0]['question_id'])
		qn2 = str(self.annotData['annotations'][1]['question_id'])
		qn1encode, qn1 = self.qnProcessor.getEncodedQn(qn1)
		qn2encode, qn2 = self.qnProcessor.getEncodedQn(qn2)
		print(qn1)
		print(qn2)

		firstVec = qn1encode + self.imgData[str(self.annotData['annotations'][0]['image_id'])][0]
		npvec = np.array(firstVec)
		#print(npvec)
		print(len(qn1encode))
		print(len(firstVec))


def encodeQn(qn, bowDimMap, bowLen):
	#return bag of words vector for the qn
	qnVec = [0] * bowLen
	for word in word_tokenize(qn.lower()):
		if (word != '?'):
			qnVec[bowDimMap[word]] = qnVec[bowDimMap[word]] + 1
	return qnVec

def encodeAns(ans, ansClassMap, ansClassLen):
	ansVec = [0] * ansClassLen
	if (ans in ansClassMap):
		ansVec[ansClassMap[ans]] = 1
	return ansVec


def readJsonFile(jsonFile):
	with open(jsonFile) as jFile:
		data = json.load(jFile)
	return data

def resolveAnswer(possibleAnswersList):
	answers = []
	for answerDetails in possibleAnswersList:
		answers.append(answerDetails['answer'])
	mostCommon = Counter(answers).most_common(1)
	return mostCommon[0][0]

def getBOWdimensions():
	#Return list of word dimensions and a map word->position dimension
	with open(vocabBOWfile, 'rb') as bowFile:
		reader = csv.reader(bowFile, delimiter=',')
		bowD = next(reader)

	index = 0
	wordDimMap = {}
	for word in bowD:
		wordDimMap[word] = index
		index = index+1 

	return bowD, wordDimMap

def get1000MostFreqAnswers():
	with open(mostFreqAnswersFile, 'rb') as ansFile:
		reader = csv.reader(ansFile, delimiter=',')
		ansVec = next(reader)

	index = 0
	ansClassMap = {}
	for word in ansVec:
		ansClassMap[word] = index
		index = index + 1 

	return ansVec, ansClassMap

if __name__ == "__main__":
	questionFile = '/media/jwong/Transcend/VQADataset/TrainSet/Questions_Train_mscoco/Preprocessed/processedOpenEnded_trainQns.json'
	annotationsFile = '/media/jwong/Transcend/VQADataset/TrainSet/mscoco_train_annotations.json'
	imageFile = '/media/jwong/Transcend/VQADataset/TrainSet/ExtractedImageFeatures/VQAImgFeatures_Train.json'
	qnData = readJsonFile(questionFile)
	imgData = readJsonFile(imageFile)

	inputProcessor = InputProcessor(questionFile, vocabBOWfile)
	readAnnotationsFile(annotationsFile, qnData, imgData)
	#get1000MostFreqAnswers()
	#getBOWdimensions()
