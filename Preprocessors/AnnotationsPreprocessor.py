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

def preprocessTrainAnnotations(annotationsFile, outFileName):
	with open(annotationsFile) as annotFile:
			annotList = json.load(annotFile)['annotations']

	start = 0
	end = 10000
	counter = 1
	while True:
		outFile = outFileName + str(counter) + '.json'
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

if __name__ == "__main__":
	annotationsTrainFile = '/media/jwong/Transcend/VQADataset/TrainSet/mscoco_train_annotations.json'
	outTrainFileName = '/media/jwong/Transcend/VQADataset/TrainSet/Minibatches_PreprocessedAnnotations/TrainMiniBatch'
	readCheck(outTrainFileName + '25.json')

	
