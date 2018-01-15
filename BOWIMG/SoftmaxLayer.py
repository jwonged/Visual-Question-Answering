import tensorflow as tf 
import numpy as np
import time
import pickle
import csv
from InputReader import InputReader

class SoftmaxLayer:
	def trainFromFile(self, trainReader, valReader):
		logs = 'BOWIMGresults.txt'
		inputVecSize = 1324 #WordVec(300) + ImgFeatures (1024)
		numOfClasses = 1000
		batchSize = 32

		#Softmax layer model
		
		x = tf.placeholder(tf.float32,[None, inputVecSize], name='x')
		w = tf.Variable(tf.zeros([inputVecSize, numOfClasses]))
		b = tf.Variable(tf.zeros([numOfClasses]))
		y = tf.matmul(x,w) + b
		
		#Loss - sparse softmax cross entropy for efficient label vector
		ylabels = tf.placeholder(tf.int64, [None], name='ylabels') #Each label is only a single int corresponding to ans class index
		crossEntropyLoss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=ylabels, logits=y))
		trainModel = tf.train.GradientDescentOptimizer(0.01).minimize(crossEntropyLoss, name='trainModel')
		
		#accuracy
		yPred = tf.argmax(tf.nn.softmax(y, name='yPred'),1) #Softmax redundant with argmax
		#is_correct_prediction = tf.equal(yPred, tf.argmax(ylabels, 1))
		is_correct_prediction = tf.equal(yPred, ylabels) #ylabels is a single value (shouldn't need argmax)
		accuracy = tf.reduce_mean(tf.cast(is_correct_prediction, tf.float32), name='accuracy')
		
		#init
		sess = tf.InteractiveSession()
		tf.global_variables_initializer().run()
		saver = tf.train.Saver()
		
		#can also print session.run(crossEntropyLoss, feed_dict=feed_dict_val)
		
		currentEpoch = 0
		
		with open(logs, 'w') as logFile:
			print('Training model...')
			while(True):
				if (trainReader.getEpoch() > currentEpoch):
					currentEpoch += 1
					print('Saving model...')
					saver.save(sess, 'BOWIMG-model')
					
					valX, valY = valReader.getWholeBatch()
					valAcc = sess.run(accuracy, feed_dict={x: valX, ylabels: valY})
					
					msg = "Epoch {0} --  Val Accuracy: {1:>6.1%}".format(trainReader.getEpoch(),valAcc)
					logFile.write(msg+'\n')
					print(msg)
				
				if (currentEpoch > 15):
					break
				
				#Train
				trainX, trainY = trainReader.getNextXYBatch(batchSize)
				sess.run(trainModel, feed_dict={x: trainX, ylabels: trainY})
	
				#Evaluate
				if (trainReader.getIndexInEpoch()%(1000*batchSize)==0):
					trainAcc = sess.run(accuracy, feed_dict={x: trainX, ylabels: trainY})
					
					valX, valY = valReader.getWholeBatch()
					valAcc = sess.run(accuracy, feed_dict={x: valX, ylabels: valY})
					
					#msg = 'Epoch_index = ' + str(trainReader.getIndexInEpoch()) + ', train accuracy = ' + str(trainAcc) + ', val accuracy = ' + str(valAcc)
					msg = "Epoch {0}, index {1} -- Train Accuracy: {2:>6.1%}, Val Accuracy: {3:>6.1%}".format(trainReader.getEpoch(), trainReader.getIndexInEpoch(), trainAcc, valAcc)
					logFile.write(msg+'\n')
					print(msg)
			print('Completed')
	
	def continueTrainingFromSavedModel(self, trainReader, valReader):
		logs = 'BOWIMGresults.txt'
		batchSize = 32
		sess = tf.Session()
		saver = tf.train.import_meta_graph('BOWIMG-model.meta')
		saver.restore(sess, tf.train.latest_checkpoint('./'))
		
		graph = tf.get_default_graph()
		
		yPred = graph.get_tensor_by_name('yPred:0')
		accuracy = graph.get_tensor_by_name('accuracy:0')
		x = graph.get_tensor_by_name('x:0')
		ylabels = graph.get_tensor_by_name('ylabels:0')
		trainModel = graph.get_operation_by_name('trainModel')
		
		saver = tf.train.Saver()
		
		currentEpoch = 0
		bestValAcc = 0
		with open(logs, 'a') as logFile:
			print('Training model...')
			while(True):
				if (trainReader.getEpoch() > currentEpoch):
					currentEpoch += 1
					print('Saving model...')
					saver.save(sess, 'BOWIMG-model')
					
					valX, valY = valReader.getWholeBatch()
					valAcc = sess.run(accuracy, feed_dict={x: valX, ylabels: valY})
					msg = "Epoch {0} --  Val Accuracy: {1:>6.1%}".format(trainReader.getEpoch()+8,valAcc)
					logFile.write(msg+'\n')
					print(msg)
					
					#early stopping
					if (bestValAcc < valAcc):
						bestValAcc = valAcc
					else:
						break
				
				if (currentEpoch > 50):
					break
				
				#Train
				trainX, trainY = trainReader.getNextXYBatch(batchSize)
				sess.run(trainModel, feed_dict={x: trainX, ylabels: trainY})
	
				#Evaluate
				if (trainReader.getIndexInEpoch()%(1000*batchSize)==0):
					trainAcc = sess.run(accuracy, feed_dict={x: trainX, ylabels: trainY})
					
					valX, valY = valReader.getWholeBatch()
					valAcc = sess.run(accuracy, feed_dict={x: valX, ylabels: valY})
					
					msg = "Epoch {0}, index {1} -- Train Accuracy: {2:>6.1%}, Val Accuracy: {3:>6.1%}".format(trainReader.getEpoch()+8, trainReader.getIndexInEpoch(), trainAcc, valAcc)
					logFile.write(msg+'\n')
		print('Completed')
	
	def testSavedModel(self, testReader):
		logs = 'BOWIMGTestresults.txt'
		batchSize = 32
		sess = tf.Session()
		saver = tf.train.import_meta_graph('BOWIMG-model.meta')
		saver.restore(sess, tf.train.latest_checkpoint('./'))
		
		graph = tf.get_default_graph()
		
		yPred = graph.get_tensor_by_name('yPred:0')
		accuracy = graph.get_tensor_by_name('accuracy:0')
		x = graph.get_tensor_by_name('x:0')
		ylabels = graph.get_tensor_by_name('ylabels:0')
		trainModel = graph.get_operation_by_name('trainModel')
		
		testX, testY = testReader.getWholeBatch()
		print('Running test...')
		testAcc = sess.run(accuracy, feed_dict={x: testX, ylabels: testY})
		msg = "Test Accuracy: {0:>6.1%}".format(testAcc)
		print(msg)
		with open(logs, 'w') as logFile:
			logFile.write(msg+'\n')
		
		print('Completed')
	
	def predictRestoredModel(self, input):
		
		ansClassMap = self.getAnsClassMap()
		sess = tf.Session()
		saver = tf.train.import_meta_graph('BOWIMG-model.meta')
		saver.restore(sess, tf.train.latest_checkpoint('./'))
		
		graph = tf.get_default_graph()
		
		yPred = graph.get_tensor_by_name('yPred')
		accuracy = graph.get_tensor_by_name('accuracy')
		x = graph.get_tensor_by_name('x')
		ylabels = graph.get_tensor_by_name('ylabels')
		
		result = sess.run(yPred, feed_dict={x: input})
		print(result)
		
	def getAnsClassMap(self):
		mostFreqAnswersFile = '/home/jwong/Documents/LinuxWorkspace/Visual-Question-Answering/resources/1000MostFreqAnswers.csv'
		
		with open(mostFreqAnswersFile, 'rb') as ansFile:
			reader = csv.reader(ansFile, delimiter=',')
			ansVec = next(reader)

		index = 0
		ansClassMap = {}
		for word in ansVec:
			ansClassMap[index] = word
			index += 1 
		return ansClassMap
		
	def trainWithProcessor(self, trainProcessor, valProcessor):
		
		#AllAns : Old-14794 #BOWDim (13770) + ImgFeatures (1024)
		inputVecSize = 1324 #WordVec(300) + ImgFeatures (1024)
		numOfClasses = 17140 #for answers -- allans=17140
		batchSize = 16

		#Softmax layer model
		x = tf.placeholder(tf.float32,[None, inputVecSize])
		w = tf.Variable(tf.zeros([inputVecSize, numOfClasses]))
		b = tf.Variable(tf.zeros([numOfClasses]))
		y = tf.matmul(x,w) + b

		#Loss
		ylabels = tf.placeholder(tf.float32, [None, numOfClasses])
		crossEntropyLoss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=ylabels, logits=y))
		trainModel = tf.train.GradientDescentOptimizer(0.01).minimize(crossEntropyLoss)
		#trainModel = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(crossEntropyLoss)
		#try adam optimizer and adadelta (speed up training / result)

		correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(ylabels, 1))
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

		#Setup
		sess = tf.InteractiveSession()
		tf.global_variables_initializer().run()
		#saver = tf.train.Saver()

		print('Training model...')
		while(True):
			if (trainProcessor.getIndexInEpoch() > 200):
				break

			trainX, trainY = trainProcessor.getNextXYBatch(batchSize)
			print('Training with batch size: ' + str(len(trainX)))
			valX, valY = valProcessor.getNextXYBatch(batchSize)
			sess.run(trainModel, feed_dict={x: trainX, ylabels: trainY})

			#Evaluate
			trainAcc = sess.run(accuracy, feed_dict={x: trainX, ylabels: trainY})
			valAcc = sess.run(accuracy, feed_dict={x: valX, ylabels: valY})

			print('Epoch_index = ' + str(trainProcessor.getIndexInEpoch()) + ', train accuracy = ' + str(trainAcc) + ', Val accuracy = ' + str(valAcc))
		print('Completed')

		'''
		#Train
		with open('Trainlog.txt', 'w') as logFile:
			for i in range(1,26):
				print('Training model... Batch: ' + str(i))
				trainX, trainY = trainProcessor.getXandYbatch(miniBatchPath + str(i) + '.json')
				sess.run(trainModel, feed_dict={x: trainX, ylabels: trainY})

				print('Evaluating model...')
				
				#test
				
				result = sess.run(accuracy, feed_dict={x: testX, ylabels: testY})
				print('Batch: ' + str(i) + ', Accuracy: ' + str(result) + '\n')
				logFile.write('Batch: ' + str(i) + ', Accuracy: ' + str(result) + '\n')
		'''

		#startT = time.time()
		#endT = time.time()

def train():
	xTrainPickle = '/media/jwong/Transcend/VQADataset/TrainSet/XYTrainData/sparseCleanWVsum1000TrainxAll.pkl'
	yTrainPickle = '/media/jwong/Transcend/VQADataset/TrainSet/XYTrainData/sparseCleanWVsum1000TrainyAll.pkl'
	
	xValPickle = '/media/jwong/Transcend/VQADataset/ValTestSet/XYValTestData/sparseCleanWVsum1000valx.pkl'
	yValPickle = '/media/jwong/Transcend/VQADataset/ValTestSet/XYValTestData/sparseCleanWVsum1000valy.pkl'
	
	xTestPickle = '/media/jwong/Transcend/VQADataset/ValTestSet/XYValTestData/sparseCleanWVsum1000testx.pkl'
	yTestPickle = '/media/jwong/Transcend/VQADataset/ValTestSet/XYValTestData/sparseCleanWVsum1000testy.pkl'
	
	#trainReader = InputReader(xTrainPickle, yTrainPickle)
	#valReader = InputReader(xValPickle, yValPickle)
	#testReader = InputReader(xTestPickle, yTestPickle)
	
	model = SoftmaxLayer()
	#model.trainFromFile(trainReader, valReader)
	#model.continueTrainingFromSavedModel(trainReader, valReader)
	#model.testSavedModel(testReader)
	
def test():
	xTestPickle = '/media/jwong/Transcend/VQADataset/ValTestSet/XYValTestData/sparseCleanWVsum1000testx.pkl'
	yTestPickle = '/media/jwong/Transcend/VQADataset/ValTestSet/XYValTestData/sparseCleanWVsum1000testy.pkl'
	
	testReader = InputReader(xTestPickle, yTestPickle)
	
	model = SoftmaxLayer()
	model.testSavedModel(testReader)

if __name__ == '__main__':
	test()


	



	
