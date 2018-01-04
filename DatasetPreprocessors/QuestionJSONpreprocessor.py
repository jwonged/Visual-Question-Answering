import json
from collections import Counter
'''
Convert questions json to map
qn_id --> qn string
'''
def processQnFile(questionFile, outFile):
	print('Reading file: ' + questionFile)

	count = 0
	questions = {}
	with open(questionFile) as jsonFile:
		qnData = json.load(jsonFile)
		for qn in qnData['questions']:
			count = count + 1
			questions[int(qn['question_id'])] = qn['question']
			if (count%1000 == 0):
				print(count)
	print('Completed: ' + str(count))

	with open(outFile, 'w') as jsonOut:
		json.dump(questions, jsonOut)
	print('Writing to: ' + outFile)

if __name__ == "__main__":

	#Train sets
	#questionFile = '/media/jwong/Transcend/VQADataset/TrainSet/Questions_Train_mscoco/OpenEnded_mscoco_train2014_questions.json'
	#outFile = '/media/jwong/Transcend/VQADataset/TrainSet/Questions_Train_mscoco/Preprocessed/processedOpenEnded_trainQns.json'
	
	#val sets
	oeQnFile = '/media/jwong/Transcend/VQADataset/ValSet/Questions_Val_mscoco/OpenEnded_mscoco_val2014_questions.json'
	oeOutFile = '/media/jwong/Transcend/VQADataset/ValSet/Questions_Val_mscoco/preprocessedValQnsOpenEnded.json'
	processQnFile(oeQnFile, oeOutFile)
