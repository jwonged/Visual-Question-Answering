import tensorflow as tf 
import numpy as np
import time

class SoftmaxLayer:
	def runSoftmaxLayer(self, inputx, inputy):
		numOfClasses = 1000 #for answer
		inputVecSize = 14794 #BOWDim (13770) + ImgFeatures (1024)

		print('Reading in batch size: ' + str(len(inputx)))
		print('Reading in label size: ' + str(len(inputy)))

		#80/20 split
		trainSize = (len(inputx)*80)/100
		trainX = inputx[:trainSize]
		trainY = inputy[:trainSize]
		testX = inputx[trainSize:]
		testY = inputy[trainSize:]

		print('Training batch size: ' + str(trainSize))

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
		print('Training model...')
		startT = time.time()
		for i in range(8):
			sess.run(trainModel, feed_dict={x: trainX, ylabels: trainY})
		endT = time.time()

		# Test trained model
		print('Evaluating model...')
		correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(ylabels, 1))
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
		print('Accuracy: ' + str(sess.run(accuracy, feed_dict={x: testX, ylabels: testY})))
		print('Time taken: ' + str(endT-startT))

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

	



	
