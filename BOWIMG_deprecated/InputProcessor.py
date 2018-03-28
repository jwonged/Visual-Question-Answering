import json
import csv
from collections import Counter
import pickle
from WVSumQuestionProcessor import WVSumQuestionProcessor

'''
Read in annotations file and process input into X and Y batch
'''

class InputProcessor:
	def __init__(self, annotFileName, questionFile, imageFile, mostFreqAnswersFile):
		print('Reading ' + imageFile)
		with open(imageFile) as jFile:
			self.imgData = json.load(jFile)
		print('Reading ' + questionFile)
		self.qnProcessor = WVSumQuestionProcessor(questionFile)
		self.mostFreqAnswersFile = mostFreqAnswersFile
		print('Reading ' + annotFileName)
		self.xlabels, self.ylabels, self.size = self.readAnnotFile(annotFileName)
		self.index = 0
		self.epoch = 0

	def getEpochSize(self):
		return self.size

	def getIndexInEpoch(self):
		return self.index

	def getEpoch(self):
		return self.epoch

	def getNextXYBatch(self, batchSize):
		if self.index > self.size: 
			#start on new epoch
			self.index = 0
			self.epoch += 1
			
		start = self.index
		self.index += batchSize
		end = self.index
		return self.xlabels[start:end], self.ylabels[start:end]

	def readAnnotFile(self, annotFileName):
		with open(annotFileName) as annotFile:
			annotBatch = json.load(annotFile)
		ansClasses, ansClassMap = self.getNMostFreqAnswers()
		ansClassLen = len(ansClasses)
		
		ylabels = []
		xlabels = []
		numOfAns = 0
		for annot in annotBatch:
			singleAns = self.resolveAnswer(annot['answers'])
			ansVec = self.encodeAns(singleAns, ansClassMap, ansClassLen)
			ylabels.append(ansVec)

			qnVec = self.qnProcessor.getEncodedQn(annot['question_id'])
			imgVec = self.imgData[str(annot['image_id'])][0]
			xVec = qnVec + imgVec
			assert len(xVec) == 1324
			xlabels.append(xVec)

			#checks
			numOfAns += 1
			if(numOfAns%(len(annotBatch)/5) == 0):
				print('Number of ans processed: ' + str(numOfAns))

		print('Batch size produced: ' + str(numOfAns))
		return xlabels, ylabels, numOfAns

	def encodeAns(self, ans, ansClassMap, ansClassLen):
		ansVec = [0] * ansClassLen
		if (ans in ansClassMap):
			ansVec[ansClassMap[ans]] = 1
		return ansVec

	def resolveAnswer(self, possibleAnswersList):
		answers = []
		for answerDetails in possibleAnswersList:
			answers.append(answerDetails['answer'])
		mostCommon = Counter(answers).most_common(1)
		return mostCommon[0][0]

	def getNMostFreqAnswers(self):
		with open(self.mostFreqAnswersFile, 'rb') as ansFile:
			reader = csv.reader(ansFile, delimiter=',')
			ansVec = next(reader)

		index = 0
		ansClassMap = {}
		for word in ansVec:
			ansClassMap[word] = index
			index = index + 1 

		return ansVec, ansClassMap
	
	def getData(self):
		return self.xlabels, self.ylabels


def pickleData(self, dataX, dataY, xfile, yfile):
	print('Writing ' + xfile)
	with open(xfile, 'wb') as pklFile:
		pickle.dump(dataX, pklFile)
	print('Writing ' + yfile)
	with open(yfile, 'wb') as pklFile:
		pickle.dump(dataY, pklFile)

if __name__ == "__main__":
	mostFreqAnswersFile = '/home/jwong/Documents/LinuxWorkspace/Visual-Question-Answering/resources/1000MostFreqAnswers.csv'
	
	print('Loading files...')
	
	xTrainPickle = '/media/jwong/Transcend/VQADataset/TrainSet/XYTrainData/WVsum1000Trainx'
	yTrainPickle = '/media/jwong/Transcend/VQADataset/TrainSet/XYTrainData/WVsum1000Trainy'
	
	trainWVQnsFile = '/media/jwong/Transcend/VQADataset/TrainSet/ExtractedQnFeatures/word2VecAddQnFeatures_Train.json'
	trainImgFile = '/media/jwong/Transcend/VQADataset/TrainSet/ExtractedImageFeatures/VQAImgFeatures_Train.json'
	trainAnnotFile = '/media/jwong/Transcend/VQADataset/TrainSet/AllTrainAnnotationsList.json'
	trainProcessor = InputProcessor(trainAnnotFile, trainWVQnsFile, trainImgFile, mostFreqAnswersFile)
	dataX, dataY = trainProcessor.getData()
	
	pickleData(dataX=dataX[], dataY=dataY[], xTrainPickle+'1.pkl', yTrainPickle+'1.pkl')
	pickleData(dataX=dataX[], dataY=dataY[], xTrainPickle+'1.pkl', yTrainPickle+'1.pkl')
	pickleData(dataX=dataX[], dataY=dataY[], xTrainPickle+'1.pkl', yTrainPickle+'1.pkl')
	#trainProcessor.pickleData(xfile=xTrainPickle, yfile=yTrainPickle)
	print('Train files completed.')
	del trainProcessor
	
	xValPickle = '/media/jwong/Transcend/VQADataset/ValTestSet/XYValTestData/WVsum1000valx.pkl'
	yValPickle = '/media/jwong/Transcend/VQADataset/ValTestSet/XYValTestData/WVsum1000valy.pkl'
	xTestPickle = '/media/jwong/Transcend/VQADataset/ValTestSet/XYValTestData/WVsum1000testx.pkl'
	yTestPickle = '/media/jwong/Transcend/VQADataset/ValTestSet/XYValTestData/WVsum1000testy.pkl'
	
	valTestWVQnsFile = '/media/jwong/Transcend/VQADataset/ValTestSet/ExtractedQnFeatures/word2VecAddQnFeatures_valTest.json'
	valImgFile = '/media/jwong/Transcend/VQADataset/ValTestSet/VQAImgFeatures_Val.json'
	valAnnotFile = '/media/jwong/Transcend/VQADataset/ValTestSet/AllValAnnotationsList.json'
	
	valProcessor = InputProcessor(valAnnotFile, valTestWVQnsFile, valImgFile, mostFreqAnswersFile)
	valProcessor.pickleData(xfile=xValPickle, yfile=yValPickle)
	print('Val files completed.')
	del valProcessor
	
	testImgFile = '/media/jwong/Transcend/VQADataset/ValTestSet/VQAImgFeatures_Test.json'
	testAnnotFile = '/media/jwong/Transcend/VQADataset/ValTestSet/AllTestAnnotationsList.json'
	valProcessor = InputProcessor(testAnnotFile, valTestWVQnsFile, testImgFile, mostFreqAnswersFile)
	valProcessor.pickleData(xfile=xTestPickle, yfile=yTestPickle)
	
	
	
