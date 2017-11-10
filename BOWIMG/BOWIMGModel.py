import InputProcessor
import SoftmaxLayer

'''
SemiEnd to end BOWIMG model (excluding preprocessing)
'''

if __name__ == "__main__":
	#files that depend on set
	questionFile = '/media/jwong/Transcend/VQADataset/TrainSet/Questions_Train_mscoco/Preprocessed/processedOpenEnded_trainQns.json'
	annotationsFile = '/media/jwong/Transcend/VQADataset/TrainSet/mscoco_train_annotations.json'
	imageFile = '/media/jwong/Transcend/VQADataset/TrainSet/ExtractedImageFeatures/VQAImgFeatures_Train.json'

	#constant files
	mostFreqAnswersFile = '/home/jwong/Documents/LinuxWorkspace/Visual-Question-Answering/resources/1000MostFreqAnswers.csv'
	vocabBOWfile = '/home/jwong/Documents/LinuxWorkspace/Visual-Question-Answering/resources/BOWdimensions.csv'
	
	print('Loading files...')
	inputProcessor = InputProcessor.InputProcessor(questionFile, vocabBOWfile, imageFile, annotationsFile, mostFreqAnswersFile)
	xVals, yVals = inputProcessor.getXandYbatch()

	Decider = SoftmaxLayer.SoftmaxLayer()
	Decider.runSoftmaxLayer(xVals, yVals)
	print('Done.')