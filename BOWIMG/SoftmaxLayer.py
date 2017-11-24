import tensorflow as tf 
import numpy as np
import time

class SoftmaxLayer:
	def runSoftmaxLayer(self, trainProcessor, testProcessor):
		
		#AllAns 
		inputVecSize = 14794 #BOWDim (13770) + ImgFeatures (1024)
		numOfClasses = 1000 #for answers -- allans=17140
		miniBatchPath = '/media/jwong/Transcend/VQADataset/TrainSet/trainMiniBatches/TrainMiniBatch'

		print('Reading test batches')
		testX, testY = testProcessor.getXandYbatch('/media/jwong/Transcend/VQADataset/ValSet/testMiniBatches/testMiniBatch1.json')

		#print('Reading in batch size: ' + str(len(trainX)))
		#print('Reading in label size: ' + str(len(trainY)))		

		#Softmax layer model
		x = tf.placeholder(tf.float32,[None, inputVecSize])
		w = tf.Variable(tf.zeros([inputVecSize, numOfClasses]))
		b = tf.Variable(tf.zeros([numOfClasses]))
		y = tf.matmul(x,w) + b

		#Loss
		ylabels = tf.placeholder(tf.float32, [None, numOfClasses])
		crossEntropyLoss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=ylabels, logits=y))
		trainModel = tf.train.GradientDescentOptimizer(0.01).minimize(crossEntropyLoss)
		#try adam optimizer and adadelta (speed up training / result)

		#Setup
		sess = tf.InteractiveSession()
		tf.global_variables_initializer().run()

		#Train
		with open('Trainlog.txt', 'w') as logFile:
			for i in range(1,26):
				print('Training model... Batch: ' + str(i))
				trainX, trainY = trainProcessor.getXandYbatch(miniBatchPath + str(i) + '.json')
				sess.run(trainModel, feed_dict={x: trainX, ylabels: trainY})

				print('Evaluating model...')
				
				#test
				correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(ylabels, 1))
				accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
				result = sess.run(accuracy, feed_dict={x: testX, ylabels: testY})
				logFile.write('Batch: ' + str(i) + ', Accuracy: ' + str(result) + '\n')

		print('Completed')
		#startT = time.time()
		#endT = time.time()

	def readNPfile(self, fileName):
		print('Reading file: ' + fileName)
		with open(fileName, 'r') as file:
			return np.loadtxt(file)


if __name__ == '__main__':

	#X Y np files
	xfile = '/media/jwong/Transcend/VQADataset/TrainSet/inputBatches/testBatchX.out'
	yfile = '/media/jwong/Transcend/VQADataset/TrainSet/inputBatches/testBatchY.out'

	Decider = SoftmaxLayer()
	inputx = Decider.readNPfile(xfile)
	inputy = Decider.readNPfile(yfile)
	Decider.runSoftmaxLayer(inputx, inputy)

	



	
