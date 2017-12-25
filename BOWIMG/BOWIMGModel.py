from InputProcessor import InputProcessor
from SoftmaxLayer import SoftmaxLayer
from InputReader import InputReader

'''
SemiEnd to end BOWIMG model (excluding preprocessing)
'''

def main():
	mostFreqAnswersFile = '/home/jwong/Documents/LinuxWorkspace/Visual-Question-Answering/resources/allTrainAnswers.csv'
	
	print('Loading files...')
	
	trainWVQnsFile = '/media/jwong/Transcend/VQADataset/TrainSet/ExtractedQnFeatures/word2VecAddQnFeatures_Train.json'
	trainImgFile = '/media/jwong/Transcend/VQADataset/TrainSet/ExtractedImageFeatures/VQAImgFeatures_Train.json'
	trainAnnotFile = '/media/jwong/Transcend/VQADataset/TrainSet/AllTrainAnnotationsList.json'
	trainProcessor = InputProcessor(trainAnnotFile, trainWVQnsFile, trainImgFile, mostFreqAnswersFile)

	valTestWVQnsFile = '/media/jwong/Transcend/VQADataset/ValTestSet/ExtractedQnFeatures/word2VecAddQnFeatures_valTest.json'
	valImgFile = '/media/jwong/Transcend/VQADataset/ValTestSet/VQAImgFeatures_Val.json'
	valAnnotFile = '/media/jwong/Transcend/VQADataset/ValTestSet/AllValAnnotationsList.json'
	testImgFile = '/media/jwong/Transcend/VQADataset/ValTestSet/VQAImgFeatures_Test.json'
	testAnnotFile = '/media/jwong/Transcend/VQADataset/ValTestSet/AllTestAnnotationsList.json'

	valProcessor = InputProcessor(valAnnotFile, valTestWVQnsFile, valImgFile, mostFreqAnswersFile)

	trainer = SoftmaxLayer()
	trainer.trainSoftmaxLayer(trainProcessor, valProcessor)
	
	#testProcessor = InputProcessor(valTestQnFile, vocabBOWfile, testImgFile, mostFreqAnswersFile)
	#testProcessor.readAnnotFile(testAnnotFile)
	print('Completed.')
	

def trainFromDataFile():
	xTrainjson = '/media/jwong/Transcend/VQADataset/TrainSet/XYTrainData/WVsum1000Trainx.json'
	yTrainjson = '/media/jwong/Transcend/VQADataset/TrainSet/XYTrainData/WVsum1000Trainy.json'
	xValjson = '/media/jwong/Transcend/VQADataset/ValTestSet/XYValTestData/WVsum1000valx.json'
	yValjson = '/media/jwong/Transcend/VQADataset/ValTestSet/XYValTestData/WVsum1000valy.json'
	xTestjson = '/media/jwong/Transcend/VQADataset/ValTestSet/XYValTestData/WVsum1000testx.json'
	yTestjson = '/media/jwong/Transcend/VQADataset/ValTestSet/XYValTestData/WVsum1000testy.json'
	miniTrainX = '/media/jwong/Transcend/VQADataset/DummySets/miniTrainX.pkl'
	miniTrainY = '/media/jwong/Transcend/VQADataset/DummySets/miniTrainY.pkl'
	miniValX = '/media/jwong/Transcend/VQADataset/DummySets/miniValX.pkl'
	miniValY = '/media/jwong/Transcend/VQADataset/DummySets/miniValY.pkl'
	
	xTrainPickle1 = '/media/jwong/Transcend/VQADataset/TrainSet/XYTrainData/WVsum1000Trainx1.pkl'
	xTrainPickle2 = '/media/jwong/Transcend/VQADataset/TrainSet/XYTrainData/WVsum1000Trainx2.pkl'
	xTrainPickle3 = '/media/jwong/Transcend/VQADataset/TrainSet/XYTrainData/WVsum1000Trainx3.pkl'

	yTrainPickle1 = '/media/jwong/Transcend/VQADataset/TrainSet/XYTrainData/WVsum1000Trainy1.pkl'
	yTrainPickle2 = '/media/jwong/Transcend/VQADataset/TrainSet/XYTrainData/WVsum1000Trainy2.pkl'
	yTrainPickle3 = '/media/jwong/Transcend/VQADataset/TrainSet/XYTrainData/WVsum1000Trainy3.pkl'
	
	xValPickle = '/media/jwong/Transcend/VQADataset/ValTestSet/XYValTestData/WVsum1000valx.pkl'
	yValPickle = '/media/jwong/Transcend/VQADataset/ValTestSet/XYValTestData/WVsum1000valy.pkl'
	xTestPickle = '/media/jwong/Transcend/VQADataset/ValTestSet/XYValTestData/WVsum1000testx.pkl'
	yTestPickle = '/media/jwong/Transcend/VQADataset/ValTestSet/XYValTestData/WVsum1000testy.pkl'
	
	
	trainReader = InputReader(xTrainPickle1, yTrainPickle1)
	valReader = InputReader(xValPickle, yValPickle)
	
	model = SoftmaxLayer()
	model.trainFromFile(trainReader, valReader)
	
if __name__ == "__main__":
	trainFromDataFile()

#Old files
#mostFreqAnswersFile = '/home/jwong/Documents/LinuxWorkspace/Visual-Question-Answering/resources/1000MostFreqAnswers.csv'
#miniBatchPath = '/media/jwong/Transcend/VQADataset/TrainSet/trainMiniBatches/TrainMiniBatch'
#trainQnsFile = '/media/jwong/Transcend/VQADataset/TrainSet/Questions_Train_mscoco/Preprocessed/processedOpenEnded_trainQns.json'
#valTestQnFile = '/media/jwong/Transcend/VQADataset/ValTestSet/Questions_Val_mscoco/preprocessedValTestQnsOpenEnded.json'
#vocabBOWfile = '/home/jwong/Documents/LinuxWorkspace/Visual-Question-Answering/resources/BOWdimensions.csv'
