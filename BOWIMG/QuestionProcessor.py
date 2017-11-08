import json
import csv
from collections import Counter
from nltk import word_tokenize

class QuestionProcessor:
	def __init__(self, questionFile, vocabBOWfile):
		with open(questionFile) as qnFile:
			qnData = json.load(qnFile)

		with open(vocabBOWfile, 'rb') as bowFile:
			reader = csv.reader(bowFile, delimiter=',')
			bowDim = next(reader)

		index = 0
		bowDimMap = {}
		for word in bowDim:
			bowDimMap[word] = index
			index = index + 1 

		self.qnData = qnData
		self.bowDim = bowDim
		self.bowDimMap = bowDimMap
		self.bowLen = len(bowDim)

	def getQn(self, qnID):
		return self.qnData[str(qnID)]

	def getEncodedQn(self, qnID):
		#return bag of words vector for the qn
		qnVec = [0] * self.bowLen
		qn = self.getQn(qnID)
		for word in word_tokenize(qn.lower()):
			if (word != '?'):
				qnVec[self.bowDimMap[word]] = qnVec[self.bowDimMap[word]] + 1
		return qnVec, qn

if __name__ == "__main__":
	questionFile = '/media/jwong/Transcend/VQADataset/TrainSet/Questions_Train_mscoco/Preprocessed/processedOpenEnded_trainQns.json'
	vocabBOWfile = '/home/jwong/Documents/LinuxWorkspace/Visual-Question-Answering/resources/BOWdimensions.csv'
	processor = QuestionProcessor(questionFile, vocabBOWfile)
	vec, qn = processor.getEncodedQn(5577940)
	print(qn)
