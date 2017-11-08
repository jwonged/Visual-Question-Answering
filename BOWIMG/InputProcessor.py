import json
import csv
import numpy as np
from collections import Counter
from nltk import word_tokenize
import QuestionProcessor
'''
Read in annotations
Retrieve answer
'''

class InputProcessor:
	def __init__(self, questionFile, vocabBOWfile, imageFile, annotationsFile, mostFreqAnswersFile):
		with open(annotationsFile) as annotFile:
			annotData = json.load(annotFile)
		self.annotData = annotData
		self.imgData = self.readJsonFile(imageFile)
		self.qnProcessor = QuestionProcessor.QuestionProcessor(questionFile, vocabBOWfile)
		self.mostFreqAnswersFile = mostFreqAnswersFile

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

	def getXandYbatch(self):
		ansClasses, ansClassMap = self.get1000MostFreqAnswers()
		ansClassLen = len(ansClasses)

		ylabels = []
		xlabels = []
		numOfAns = 0
		for annot in self.annotData['annotations']:
			singleAns = self.resolveAnswer(annot['answers'])
			ansVec = self.encodeAns(singleAns, ansClassMap, ansClassLen)
			ylabels.append(ansVec)

			qnVec, qn = self.qnProcessor.getEncodedQn(annot['question_id'])
			print('Processing:' + qn)
			xVec = qnVec + self.imgData[str(annot['image_id'])][0]
			xlabels.append(xVec)
			numOfAns = numOfAns + 1
			if (numOfAns > 10):
				break

		return xlabels, ylabels

	def encodeAns(self, ans, ansClassMap, ansClassLen):
		ansVec = [0] * ansClassLen
		if (ans in ansClassMap):
			ansVec[ansClassMap[ans]] = 1
		return ansVec

	def readJsonFile(self, jsonFile):
		with open(jsonFile) as jFile:
			data = json.load(jFile)
		return data

	def resolveAnswer(self, possibleAnswersList):
		answers = []
		for answerDetails in possibleAnswersList:
			answers.append(answerDetails['answer'])
		mostCommon = Counter(answers).most_common(1)
		return mostCommon[0][0]

	def get1000MostFreqAnswers(self):
		with open(self.mostFreqAnswersFile, 'rb') as ansFile:
			reader = csv.reader(ansFile, delimiter=',')
			ansVec = next(reader)

		index = 0
		ansClassMap = {}
		for word in ansVec:
			ansClassMap[word] = index
			index = index + 1 

		return ansVec, ansClassMap

if __name__ == "__main__":
	#files that depend on set
	questionFile = '/media/jwong/Transcend/VQADataset/TrainSet/Questions_Train_mscoco/Preprocessed/processedOpenEnded_trainQns.json'
	annotationsFile = '/media/jwong/Transcend/VQADataset/TrainSet/mscoco_train_annotations.json'
	imageFile = '/media/jwong/Transcend/VQADataset/TrainSet/ExtractedImageFeatures/VQAImgFeatures_Train.json'
	
	#constant files
	mostFreqAnswersFile = '/home/jwong/Documents/LinuxWorkspace/Visual-Question-Answering/resources/1000MostFreqAnswers.csv'
	vocabBOWfile = '/home/jwong/Documents/LinuxWorkspace/Visual-Question-Answering/resources/BOWdimensions.csv'
	
	inputProcessor = InputProcessor(questionFile, vocabBOWfile, imageFile, annotationsFile, mostFreqAnswersFile)
	xlabels, ylabels = inputProcessor.getXandYbatch()
	print(xlabels[0])
	print(ylabels[0])
	print(len(xlabels))
	print(len(xlabels[0]))
	print(len(ylabels))
	print(len(ylabels[0]))
