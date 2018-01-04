import json
from collections import Counter
import csv
'''
Output a file containing the N most frequent answers
'''
def readAnnotationsFile(annotationsFile):
	with open(annotationsFile) as annotFile:
		print('Reading from ' + annotationsFile)
		annotData = json.load(annotFile)

	allAnswers = []
	count = 0
	for annot in annotData['annotations']:
		ans = resolveAnswer(annot['answers'])
		allAnswers.append(ans)
		count = count + 1
		if (count%1000 == 0):
			print(count)
	print('Completed processing: ' + str(count))

	return allAnswers

def getMostFreqAnswers(annotationsFile, numOfAns):
	allAnswers = readAnnotationsFile(annotationsFile)
	mostFreqAnswers = Counter(allAnswers).most_common(numOfAns) #tuple
	cleanedMostFreqAnswers = []
	for answer in mostFreqAnswers:
		cleanedMostFreqAnswers.append(answer[0])
	return cleanedMostFreqAnswers #list

def getAllAnswers(annotationsFile):
	allAnswers = readAnnotationsFile(annotationsFile)
	answerSet = set()
	answerList = []
	numOfItems = 0
	for answer in allAnswers:
		if answer not in answerSet:
			answerSet.add(answer)
			answerList.append(answer)
			numOfItems = numOfItems + 1
	print('Number of items: ' + str(numOfItems))
	return answerList

def resolveAnswer(possibleAnswersList):
	#Majority vote on the 10 possible answers
	answers = []
	for answerDetails in possibleAnswersList:
		answers.append(answerDetails['answer'])
	mostCommon = Counter(answers).most_common(1)
	return mostCommon[0][0]

def writeToFile(fileName, mostFreqAnswers):
	print('Writing to file: ' + fileName)
	with open(fileName, 'w') as outputFile:
		writer = csv.writer(outputFile)
		writer.writerow(mostFreqAnswers)
	print('Writing to file: Done')


if __name__ == "__main__":
	annotationsFile = '/media/jwong/Transcend/VQADataset/TrainSet/mscoco_train_annotations.json'
	
	#N most freq answers
	outputFile = '/media/jwong/Transcend/VQADataset/TrainSet/1000MostFreqAnswers.csv'
	#thousandMostFreq = getMostFreqAnswers(annotationsFile, 1000)
	#writeToFile(outputFile, thousandMostFreq)

	#GetAllAnswers
	allAnsOutputFile = '/media/jwong/Transcend/VQADataset/TrainSet/allTrainAnswers.csv'
	writeToFile(allAnsOutputFile, getAllAnswers(annotationsFile))


