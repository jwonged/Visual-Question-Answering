import tensorflow as tf 
import numpy as np
import time
import pickle

class SoftmaxLayer:
	def trainFromFile(self, trainReader, valReader):
		inputVecSize = 1324 #WordVec(300) + ImgFeatures (1024)
		numOfClasses = 1000
		batchSize = 32

		#Softmax layer model
		x = tf.placeholder(tf.float32,[None, inputVecSize])
		w = tf.Variable(tf.zeros([inputVecSize, numOfClasses]))
		b = tf.Variable(tf.zeros([numOfClasses]))
		y = tf.matmul(x,w) + b
		
		#Loss - sparse softmax cross entropy for efficient label vector
		ylabels = tf.placeholder(tf.float32, [None, numOfClasses])
		crossEntropyLoss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=ylabels, logits=y))
		trainModel = tf.train.GradientDescentOptimizer(0.01).minimize(crossEntropyLoss)
		
		#accuracy
		yPred = tf.argmax(tf.nn.softmax(y), 1) #leaving redundant softmax on y
		correct_prediction = tf.equal(yPred, tf.argmax(ylabels, 1))
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
		
		#init
		sess = tf.InteractiveSession()
		tf.global_variables_initializer().run()
		#saver = tf.train.Saver()
		
		print('Training model...')
		while(True):
			if (trainReader.getEpoch() > 1):
				break

			trainX, trainY = trainReader.getNextXYBatch(batchSize)
			print('Training with batch size: {}'.format(len(trainX)))
			
			sess.run(trainModel, feed_dict={x: trainX, ylabels: trainY})

			#Evaluate
			if (trainReader.getIndexInEpoch()%(200*batchSize)==0):#%(trainReader.getEpochSize/batchSize *batchSize*20)==0):
				trainAcc = sess.run(accuracy, feed_dict={x: trainX, ylabels: trainY})
				
				#if ((trainReader.getIndexInEpoch() > trainReader.getEpochSize()/2) 
				#	and (trainReader.getIndexInEpoch() > (trainReader.getEpochSize()/2 + batchSize + 2))):
				valX, valY = valReader.getWholeBatch()
				valAcc = sess.run(accuracy, feed_dict={x: valX, ylabels: valY})
				#print('Epoch_index = ' + str(trainReader.getIndexInEpoch()) + ', Val accuracy = ' + str(valAcc))
			
	
				print('Epoch_index = ' + str(trainReader.getIndexInEpoch()) + ', train accuracy = ' + str(trainAcc) + ', val accuracy = ' + str(valAcc))
		print('Completed')
		
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


if __name__ == '__main__':
	xTrainPickle = '/media/jwong/Transcend/VQADataset/TrainSet/XYTrainData/WVsum1000Trainx.pkl'
	yTrainPickle = '/media/jwong/Transcend/VQADataset/TrainSet/XYTrainData/WVsum1000Trainy.pkl'
	xValPickle = '/media/jwong/Transcend/VQADataset/ValTestSet/XYValTestData/WVsum1000valx.pkl'
	yValPickle = '/media/jwong/Transcend/VQADataset/ValTestSet/XYValTestData/WVsum1000valy.pkl'
	xTestPickle = '/media/jwong/Transcend/VQADataset/ValTestSet/XYValTestData/WVsum1000testx.pkl'
	yTestPickle = '/media/jwong/Transcend/VQADataset/ValTestSet/XYValTestData/WVsum1000testy.pkl'
	
	with open(xTrainPickle, 'rb') as pklFile:
		trainX = pickle.load(pklFile)
	with open(yTrainPickle, 'rb') as pklFile:
		trainY = pickle.load(pklFile)
	


	



	
