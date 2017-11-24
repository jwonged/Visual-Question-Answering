import InputProcessor
import SoftmaxLayer

'''
SemiEnd to end BOWIMG model (excluding preprocessing)
'''

if __name__ == "__main__":
	#files that depend on set
	
	#constant files
	mostFreqAnswersFile = '/home/jwong/Documents/LinuxWorkspace/Visual-Question-Answering/resources/1000MostFreqAnswers.csv'
	vocabBOWfile = '/home/jwong/Documents/LinuxWorkspace/Visual-Question-Answering/resources/BOWdimensions.csv'

	#1-25 train batches
	#miniBatchPath = '/media/jwong/Transcend/VQADataset/TrainSet/trainMiniBatches/TrainMiniBatch'
	print('Loading files...')
	
	questionFile = '/media/jwong/Transcend/VQADataset/TrainSet/Questions_Train_mscoco/Preprocessed/processedOpenEnded_trainQns.json'
	imageFile = '/media/jwong/Transcend/VQADataset/TrainSet/ExtractedImageFeatures/VQAImgFeatures_Train.json'
	trainProcessor = InputProcessor.InputProcessor(questionFile, vocabBOWfile, imageFile, mostFreqAnswersFile)

	testQnFile = '/media/jwong/Transcend/VQADataset/ValSet/Questions_Val_mscoco/preprocessedValQnsOpenEnded.json'
	testImgFile = '/media/jwong/Transcend/VQADataset/ValSet/VQAImgFeatures_Test.json'
	testProcessor = InputProcessor.InputProcessor(testQnFile, vocabBOWfile, testImgFile, mostFreqAnswersFile)
	
	Decider = SoftmaxLayer.SoftmaxLayer()
	Decider.runSoftmaxLayer(trainProcessor, testProcessor)
	print('Done.')