from InputProcessor import InputProcessor
import SoftmaxLayer

'''
SemiEnd to end BOWIMG model (excluding preprocessing)
'''

def main():
	#constant files
	newtrainOut = '/media/jwong/Transcend/VQADataset/TrainSet/ExtractedQnFeatures/word2VecAddQnFeatures_Train.json'
	newvalTestOut = '/media/jwong/Transcend/VQADataset/ValTestSet/ExtractedQnFeatures/word2VecAddQnFeatures_valTest.json'
	
	mostFreqAnswersFile = '/home/jwong/Documents/LinuxWorkspace/Visual-Question-Answering/resources/allTrainAnswers.csv'
	vocabBOWfile = '/home/jwong/Documents/LinuxWorkspace/Visual-Question-Answering/resources/BOWdimensions.csv'

	#1-25 train batches
	
	print('Loading files...')
	
	trainQnsFile = '/media/jwong/Transcend/VQADataset/TrainSet/Questions_Train_mscoco/Preprocessed/processedOpenEnded_trainQns.json'
	trainImgFile = '/media/jwong/Transcend/VQADataset/TrainSet/ExtractedImageFeatures/VQAImgFeatures_Train.json'
	trainAnnotFile = '/media/jwong/Transcend/VQADataset/TrainSet/AllTrainAnnotationsList.json'
	trainProcessor = InputProcessor(trainQnsFile, vocabBOWfile, trainImgFile, mostFreqAnswersFile)
	trainProcessor.readAnnotFile(trainAnnotFile)

	valTestQnFile = '/media/jwong/Transcend/VQADataset/ValTestSet/Questions_Val_mscoco/preprocessedValTestQnsOpenEnded.json'
	valImgFile = '/media/jwong/Transcend/VQADataset/ValTestSet/VQAImgFeatures_Val.json'
	valAnnotFile = '/media/jwong/Transcend/VQADataset/ValTestSet/AllValAnnotationsList.json'
	testImgFile = '/media/jwong/Transcend/VQADataset/ValTestSet/VQAImgFeatures_Test.json'
	testAnnotFile = '/media/jwong/Transcend/VQADataset/ValTestSet/AllTestAnnotationsList.json'

	valProcessor = InputProcessor(valTestQnFile, vocabBOWfile, valImgFile, mostFreqAnswersFile)
	valProcessor.readAnnotFile(valAnnotFile)

	trainer = SoftmaxLayer()
	trainer.trainSoftmaxLayer(trainProcessor, valProcessor)

	#testProcessor = InputProcessor(valTestQnFile, vocabBOWfile, testImgFile, mostFreqAnswersFile)
	#testProcessor.readAnnotFile(testAnnotFile)
	print('Completed.')


if __name__ == "__main__":
	#files that depend on set
	main()
	

#Old files
#mostFreqAnswersFile = '/home/jwong/Documents/LinuxWorkspace/Visual-Question-Answering/resources/1000MostFreqAnswers.csv'
#miniBatchPath = '/media/jwong/Transcend/VQADataset/TrainSet/trainMiniBatches/TrainMiniBatch'