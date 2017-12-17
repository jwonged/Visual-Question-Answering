import json
import csv
from nltk import word_tokenize
from abc import ABCMeta

class QuestionProcessor():
	__metaclass__ = ABCMeta

if __name__ == "__main__":
	questionFile = '/media/jwong/Transcend/VQADataset/TrainSet/Questions_Train_mscoco/Preprocessed/processedOpenEnded_trainQns.json'
	vocabBOWfile = '/home/jwong/Documents/LinuxWorkspace/Visual-Question-Answering/resources/BOWdimensions.csv'
	processor = QuestionProcessor(questionFile, vocabBOWfile)
	vec, qn = processor.getEncodedQn(5577940)
	print(qn)
