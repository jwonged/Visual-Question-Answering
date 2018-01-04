#Preprocess annotations file into batches
'''
[
	{
		"question_id" : int,
		"image_id" : int,
		"question_type" : str,
		"answer_type" : str,
		"answers" : [answer],
		"multiple_choice_answer" : str
	}, ...
]
'''
import json

def preprocessTrainAnnotations():
	annotationsTrainFile = '/media/jwong/Transcend/VQADataset/TrainSet/mscoco_train_annotations.json'
	outTrainFileName = '/media/jwong/Transcend/VQADataset/TrainSet/Minibatches_PreprocessedAnnotations/TrainMiniBatch'
	
	with open(annotationsTrainFile) as annotFile:
			annotList = json.load(annotFile)['annotations']

	start = 0
	end = 10000
	counter = 1
	while True:
		outFile = outTrainFileName + str(counter) + '.json'
		print('Processing: ' + str(start) + ' to ' + str(end))
		print('Writing to file: ' + outFile)
		with open(outFile, 'w') as jsonOut:
			json.dump(annotList[start:end], jsonOut)

		start = start + 10000
		end = end + 10000
		counter = counter + 1
		
		if (start > len(annotList)):
			break

def preprocessValTestAnnotations():
	valFeaturesFile = '/media/jwong/Transcend/VQADataset/ValSet/VQAImgFeatures_Val.json'
	annotationsFile = '/media/jwong/Transcend/VQADataset/ValSet/mscoco_val2014_annotations.json'
	testFeaturesFile = '/media/jwong/Transcend/VQADataset/ValSet/VQAImgFeatures_Test.json'

	testOutPath = '/media/jwong/Transcend/VQADataset/ValSet/testMiniBatches/testMiniBatch'
	valOutPath = '/media/jwong/Transcend/VQADataset/ValSet/valMiniBatches/valMiniBatch'
	
	with open(annotationsFile) as annotFile:
		annotList = json.load(annotFile)['annotations']
	
	with open(valFeaturesFile) as valFile:
		valFeatures = json.load(valFile)

	with open(testFeaturesFile) as testFile:
		testFeatures = json.load(testFile)

	valAnnotList = []
	testAnnotList = []
	errorMessages = []

	for annot in annotList:
		img_id = str(annot['image_id'])
		if img_id in valFeatures:
			valAnnotList.append(annot)
		elif img_id in testFeatures:
			testAnnotList.append(annot)
		else:
			print('Error could not find ' + str(annot))
			errorMessages.append(str(annot))

	batchSet(valAnnotList, valOutPath)
	batchSet(testAnnotList, testOutPath)

	print('Items in annot: ' + str(len(annotList)))
	print('Items in val: ' + str(len(valAnnotList)))
	print('Items in test: ' + str(len(testAnnotList)))

	errorMessages.append('Items in annot: ' + str(len(annotList)))
	errorMessages.append('Items in val: ' + str(len(valAnnotList)))
	errorMessages.append('Items in test: ' + str(len(testAnnotList)))

	with open('Errorlog.txt', 'w') as logFile:
		for msg in errorMessages:
			logFile.write(msg)
	
def batchSet(annotList, outPath):
	start = 0
	end = 10000
	counter = 1
	while True:
		outFile = outPath + str(counter) + '.json'
		print('Processing: ' + str(start) + ' to ' + str(end))
		print('Writing to file: ' + outFile)
		with open(outFile, 'w') as jsonOut:
			json.dump(annotList[start:end], jsonOut)

		start = start + 10000
		end = end + 10000
		counter = counter + 1
		
		if (start > len(annotList)):
			break

def readCheck(outFile):
	with open(outFile) as file:
			annotData = json.load(file)

def makeDummySet():
	fileTrainPath = '/media/jwong/Transcend/VQADataset/TrainSet/trainMiniBatches/'
	fileTrainName = 'TrainMiniBatch'
	

def combineFiles():
	fileTrainPath = '/media/jwong/Transcend/VQADataset/TrainSet/trainMiniBatches/'
	fileTrainName = 'TrainMiniBatch'
	fileValPath = '/media/jwong/Transcend/VQADataset/ValSet/valMiniBatches/'
	fileValName = 'valMiniBatch'
	fileTestPath = '/media/jwong/Transcend/VQADataset/ValSet/testMiniBatches/'
	fileTestName = 'testMiniBatch'

	#check -- can append listtT?
	fullAnnotList = []
	for i in range(1,26):
		with open(fileTrainPath + fileTrainName + str(i) + '.json') as file:
			batchData = json.load(file)
			fullAnnotList = fullAnnotList + batchData

	with open(fileTrainPath + 'AllTrainAnnotationsList.json', 'w') as outFile:
		json.dump(fullAnnotList, outFile)

	fullValList = []
	for i in range(1,8):
		with open(fileValPath + fileValName + str(i) + '.json') as file:
			batchData = json.load(file)
			fullValList = fullValList + batchData

	with open(fileValPath + 'AllValAnnotationsList.json', 'w') as outFile:
		json.dump(fullValList, outFile)

	fullValList = []
	for i in range(1,8):
		with open(fileTestPath + fileTestName + str(i) + '.json') as file:
			batchData = json.load(file)
			fullValList = fullValList + batchData

	with open(fileTestPath + 'AllTestAnnotationsList.json', 'w') as outFile:
		json.dump(fullValList, outFile)


if __name__ == "__main__":
	makeDummySet()

	
