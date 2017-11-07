import json
from collections import Counter
'''
Convert questions json to map
qn_id --> qn string
'''
def processQnFile(questionFile, outFile):
	count = 0
	questions = {}
	with open(questionFile) as jsonFile:
		qnData = json.load(jsonFile)
		for qn in qnData['questions']:
			count = count + 1
			questions[qn['question_id']] = qn['question']
			if (count%100 == 0):
				print(count)
	print('Completed: ' + str(count))

	with open(outFile, 'w') as jsonOut:
		json.dump(questions, jsonOut)

if __name__ == "__main__":
	questionFile = '/media/jwong/Transcend/VQADataset/TrainSet/Questions_Train_mscoco/OpenEnded_mscoco_train2014_questions.json'
	outFile = '/media/jwong/Transcend/VQADataset/TrainSet/Questions_Train_mscoco/Preprocessed/processedOpenEnded_trainQns.json'
	processQnFile(questionFile, outFile)
