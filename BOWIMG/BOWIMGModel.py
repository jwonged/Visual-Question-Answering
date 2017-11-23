import InputProcessor
import SoftmaxLayer

'''
SemiEnd to end BOWIMG model (excluding preprocessing)
'''

if __name__ == "__main__":
	#files that depend on set
	questionFile = '/media/jwong/Transcend/VQADataset/TrainSet/Questions_Train_mscoco/Preprocessed/processedOpenEnded_trainQns.json'
	imageFile = '/media/jwong/Transcend/VQADataset/TrainSet/ExtractedImageFeatures/VQAImgFeatures_Train.json'

	#constant files
	mostFreqAnswersFile = '/home/jwong/Documents/LinuxWorkspace/Visual-Question-Answering/resources/1000MostFreqAnswers.csv'
	vocabBOWfile = '/home/jwong/Documents/LinuxWorkspace/Visual-Question-Answering/resources/BOWdimensions.csv'

	#1-25 train batches
	miniBatchPath = '/media/jwong/Transcend/VQADataset/TrainSet/trainMiniBatches/TrainMiniBatch'
	print('Loading files...')
	inputProcessor = InputProcessor.InputProcessor(questionFile, vocabBOWfile, imageFile, mostFreqAnswersFile)
	xVals, yVals = inputProcessor.getXandYbatch(miniBatchPath + str(1) + '.json')
	xtes, ytes = inputProcessor.getXandYbatch(miniBatchPath + str(2) + '.json')

	#for i in range(25):

	Decider = SoftmaxLayer.SoftmaxLayer()
	Decider.runSoftmaxLayer(xVals, yVals, xtes, ytes)
	print('Done.')