import json
import csv
from collections import Counter
from nltk import word_tokenize
'''
Read in annotations
Retrieve answer
'''

mostFreqAnswersFile = '/home/jwong/Documents/LinuxWorkspace/Visual-Question-Answering/resources/1000MostFreqAnswers.csv'
vocabBOWfile = '/home/jwong/Documents/LinuxWorkspace/Visual-Question-Answering/resources/BOWdimensions.csv'

def readAnnotationsFile(annotationsFile, qnData, imageFile):
	bowDim, bowDimMap = getBOWdimensions()
	bowLen = len(bowDim)
	ansClasses, ansClassMap = get1000MostFreqAnswers()
	ansClassLen = len(ansClasses)

	with open(annotationsFile) as annotFile:
		annotData = json.load(annotFile)

	#Get answer
	ans1 = resolveAnswer(annotData['annotations'][0]['answers'])
	ans2 = resolveAnswer(annotData['annotations'][1]['answers'])

	#Get qn
	qn1 = qnData[str(annotData['annotations'][0]['question_id'])]
	qn2 = qnData[str(annotData['annotations'][1]['question_id'])]

	print(qn1)
	print(encodeQn(qn1, bowDimMap, bowLen))
	print(qn2)
	print(encodeQn(qn2, bowDimMap, bowLen))
	print(ans1)
	print(encodeAns(ans1, ansClassMap, ansClassLen))
	print(ans2)
	print(encodeAns(ans2, ansClassMap, ansClassLen))


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


def readQnFile(questionFile):
	with open(questionFile) as qnFile:
		qnData = json.load(qnFile)
	return qnData

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
	qnData = readQnFile(questionFile)
	readAnnotationsFile(annotationsFile, qnData, imageFile)
	#get1000MostFreqAnswers()
	#getBOWdimensions()