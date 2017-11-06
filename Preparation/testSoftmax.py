import tensorflow as tf 

def runSoftmaxLayer(inputx, inputy):
	numOfClasses = 2

	#Softmax layer model
	x = tf.placeholder(tf.float32,[None, 1024])
	w = tf.variable(tf.zeroes([1024, numOfClasses]))
	b = tf.variable(tf.zeroes([numOfClasses]))
	y = tf.matmul(x,w) + b

	#Loss
	ylabels = tf.placeholder(tf.float32, [None, numOfClasses])
	crossEntropyLoss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=ylabels, logits=y))
	trainModel = tf.train.GradientDescentOptimizer(0.5).minimize(crossEntropyLoss)
	#try adam optimizer and adadelta (speed up training / result)

	#Setup
	sess = tf.InteractiveSession()
	tf.global_variables_initializer().run()

	#Train - feed_dict takes numpy arrays
	#for i in range(1000):
	#	batch_xs, batch_ys = mnist.train.next_batch(100)
	#	sess.run(trainModel, feed_dict={x: batch_xs, y_: batch_ys})

	#Train
	sess.run(trainModel, feed_dict={x: inputx, ylabels: inputy})

if __name__ == '__main__':
	main()