import json
import csv
import numpy as np
from collections import Counter
from nltk import word_tokenize
import QuestionProcessor

'''
Read in annotations file and process input into X and Y batch
'''

class InputProcessor:
	def __init__(self, questionFile, vocabBOWfile, imageFile, annotationsFile, mostFreqAnswersFile):
		with open(annotationsFile) as annotFile:
			annotData = json.load(annotFile)
		self.annotData = annotData
		self.imgData = self.readJsonFile(imageFile)
		self.qnProcessor = QuestionProcessor.QuestionProcessor(questionFile, vocabBOWfile)
		self.mostFreqAnswersFile = mostFreqAnswersFile

	def getXandYbatch(self):
		ansClasses, ansClassMap = self.get1000MostFreqAnswers()
		ansClassLen = len(ansClasses)

		batchSize = 10000

		ylabels = []
		xlabels = []
		numOfAns = 0
		for annot in self.annotData['annotations']:
			singleAns = self.resolveAnswer(annot['answers'])
			ansVec = self.encodeAns(singleAns, ansClassMap, ansClassLen)
			ylabels.append(ansVec)

			qnVec, qn = self.qnProcessor.getEncodedQn(annot['question_id'])
			#print('Processing:' + qn)
			xVec = qnVec + self.imgData[str(annot['image_id'])][0]
			xlabels.append(xVec)
			numOfAns = numOfAns + 1
			if(numOfAns%100 == 0):
				print('Number of ans processed: ' + str(numOfAns))
			if (numOfAns == batchSize):
				break

		print('Batch size produced: ' + str(numOfAns))
		return xlabels, ylabels

	def writeToCSVfile(self, fileName, data):
		with open(fileName, 'w') as csvFile:
			writer = csv.writer(csvFile)
			for item in data:
				writer.writerow(item)

	def writeToNPfile(self, fileName, data):
		with open(fileName, 'w') as (outFile):
			np.savetxt(fileName, data, fmt='%.8g')
		print('Written to: ' + fileName)

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
	
	#csv output files
	csvOutputX = '/media/jwong/Transcend/VQADataset/TrainSet/inputBatches/testBatchX.csv'
	csvOutputY = '/media/jwong/Transcend/VQADataset/TrainSet/inputBatches/testBatchY.csv'

	#np output files
	npOutputX = '/media/jwong/Transcend/VQADataset/TrainSet/inputBatches/testBatchX.out'
	npOutputY = '/media/jwong/Transcend/VQADataset/TrainSet/inputBatches/testBatchY.out'

	#constant files
	mostFreqAnswersFile = '/home/jwong/Documents/LinuxWorkspace/Visual-Question-Answering/resources/1000MostFreqAnswers.csv'
	vocabBOWfile = '/home/jwong/Documents/LinuxWorkspace/Visual-Question-Answering/resources/BOWdimensions.csv'
	
	print('Loading files...')
	inputProcessor = InputProcessor(questionFile, vocabBOWfile, imageFile, annotationsFile, mostFreqAnswersFile)
	print('Files loaded.')
	xVals, yVals = inputProcessor.getXandYbatch()
	print('xVals: ' + str(len(xVals)))
	print('yVals: ' + str(len(yVals)))
	inputProcessor.writeToNPfile(npOutputX, xVals)
	inputProcessor.writeToNPfile(npOutputY, yVals)
