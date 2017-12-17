import csv

with open('/media/jwong/Transcend/VQADataset/QuestionVocab/MultipleChoiceVocab.csv', 'rb') as csvIn1:
	with open('/media/jwong/Transcend/VQADataset/QuestionVocab/OpenEndedVocab.csv', 'rb') as csvIn2:
		with open('/media/jwong/Transcend/VQADataset/QuestionVocab/VQAVocab.csv', 'w') as csvOut:

			reader1 = csv.reader(csvIn1, delimiter=',')
			for row1 in reader1:
				vocab = row1
				print('vocab 1 : ' + str(len(vocab)))
				vocabSet = set(vocab)



			reader2 = csv.reader(csvIn2, delimiter=',')
			for row2 in reader2:
				vocab2 = row2
				print('vocab 2 : ' + str(len(vocab2)))
				for word in row2:
					if word not in vocabSet:
						vocab.append(word)

			print('vocab total : ' + str(len(vocab)))
			#writer = csv.writer(csvOut)
			#writer.writerow(vocab)