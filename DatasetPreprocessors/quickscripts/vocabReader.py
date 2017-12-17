import csv

with open('/media/jwong/Transcend/VQADataset/QuestionVocab/VQAVocab.csv', 'rb') as csvfile:
	reader = csv.reader(csvfile, delimiter=',', quotechar='|')
	for row in reader:
		print(len(row))

