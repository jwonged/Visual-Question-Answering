'''
Created on 15 Jan 2018

@author: jwong
'''

import json
import numpy as np
import tensorflow as tf 
import numpy as np
import pickle
import os

from HelperFunctions import getPretrainedw2v

class LSTMIMGmodel(object):
    '''
    Uses Bi-LSTM to a fully connected layer
    '''

    def __init__(self, config):
        self.config = config
        self.logFile = open(config.logFile, 'w')
        self.logFile.write('Initializing LSTMIMG\n')
        self.sess   = None
        self.saver  = None

    def construct(self):
        #add placeholders
        self.logFile.write('Constructing model...\n')
        # shape = (batch size, max length of sentence in batch)
        self.word_ids = tf.placeholder(tf.int32, shape=[None, None], name="word_ids")
        
        # shape = (batch size, length of image feature vector)
        self.img_vecs = tf.placeholder(tf.float32, 
                                       shape=[None, self.config.imgVecSize], 
                                       name="img_vecs")

        # shape = (batch size)
        self.sequence_lengths = tf.placeholder(tf.int32, shape=[None], name="sequence_lengths")

        # shape = (batch size, cancel(max length of sentence in batch))
        self.labels = tf.placeholder(tf.int32, shape=[None], name="labels")

        # hyper parameters
        self.dropout = tf.placeholder(dtype=tf.float32, shape=[], name="dropout")
        self.lr = tf.placeholder(dtype=tf.float32, shape=[], name="lr")

        
        #add word embeddings
        pretrainedEmbeddings = getPretrainedw2v(self.config.shortenedEmbeddingsFile)
        
        with tf.variable_scope("words"):
            wordEmbedsVar = tf.Variable(pretrainedEmbeddings,
                    name="wordEmbedsVar",
                    dtype=tf.float32,
                    trainable=self.config.trainEmbeddings)
        
        #embedding matrix, word_ids
        self.word_embeddings = tf.nn.embedding_lookup(wordEmbedsVar,
                self.word_ids, name="word_embeddings")
        
        #self.word_embeddings =  tf.nn.dropout(word_embeddings, self.dropout)
        
        
        #Handle input according to model structure
        if self.config.modelStruct == 'imagePerWord':
            #(dim of input to each LSTM cell)
            LSTM_num_units = self.config.wordVecSize + self.config.imgVecSize 
            self.LSTMinput = tf.concat([self.word_embeddings, self.img_vecs])
        else:
            LSTM_num_units = self.config.wordVecSize 
            self.LSTMinput = self.word_embeddings
            
            
        #add logits
        with tf.variable_scope("bi-lstm"):
            cell_fw = tf.contrib.rnn.LSTMCell(LSTM_num_units)
            cell_bw = tf.contrib.rnn.LSTMCell(LSTM_num_units)
            
            #fw, bw, inputs, seq len
            #checked: shape of fw_out = [?, ?, 2*LSTM_num_units]
            #Out [batch_size, max_time, cell_output_size] output, outputState
            (fw_out, bw_out), (fw_state, bw_state) = tf.nn.bidirectional_dynamic_rnn(
                cell_fw, cell_bw, 
                self.word_embeddings, 
                sequence_length=self.sequence_lengths, dtype=tf.float32)
            lstmOutput = tf.concat([fw_state.c, bw_state.c], axis=-1)
            
            nnOutput = tf.nn.dropout(lstmOutput, self.dropout)
            
        #Handle output according to model structure
        if self.config.modelStruct == 'imagePerWord':
            self.LSTMOutput = nnOutput
            LSTMOutputSize = 2*LSTM_num_units
        else:
            self.LSTMOutput = tf.concat([nnOutput, self.img_vecs], axis=-1)
            LSTMOutputSize = 2*LSTM_num_units + self.config.imgVecSize #img=[?,1024]
        
        #fully connected layer
        with tf.variable_scope("proj"):
            fchLayer = self._fullyConnectedLayer(x=self.LSTMOutput,
                                inputSize=LSTMOutputSize, 
                                outputSize=LSTMOutputSize/2, 
                                use_sig=True, num=1)
            
            y = self._fullyConnectedLayer(x=fchLayer,
                                inputSize=LSTMOutputSize/2, 
                                outputSize=self.config.nOutClasses, 
                                use_sig=False, num=2)
            
        #predict & get accuracy
        self.labels_pred = tf.cast(tf.argmax(tf.nn.softmax(y), axis=1), tf.int32, name='labels_pred')
        is_correct_prediction = tf.equal(self.labels_pred, self.labels)
        self.accuracy = tf.reduce_mean(tf.cast(is_correct_prediction, tf.float32), name='accuracy')
        
        #define losses
        crossEntropyLoss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=y, labels=self.labels)
        #mask = tf.sequence_mask(self.sequence_lengths)
        #losses = tf.boolean_mask(crossEntropyLoss, mask)
        self.loss = tf.reduce_mean(crossEntropyLoss)

        # for tensorboard
        #tf.summary.scalar("loss", self.loss)
        
        #train optimizer
        with tf.variable_scope("train_step"):
            if self.config.modelOptimizer == 'adam': 
                optimizer = tf.train.AdamOptimizer(self.lr)
            elif self.config.modelOptimizer == 'adagrad':
                optimizer = tf.train.AdagradOptimizer(self.lr)
            else:
                optimizer = tf.train.GradientDescentOptimizer(self.lr)
                
            if self.config.max_gradient_norm > 0: # gradient clipping if clip is positive
                grads, vs     = zip(*optimizer.compute_gradients(self.loss))
                grads, gnorm  = tf.clip_by_global_norm(grads, self.config.max_gradient_norm)
                self.train_op = optimizer.apply_gradients(zip(grads, vs), name='trainModel')
            else:
                self.train_op = optimizer.minimize(self.loss, name='trainModel')
                
        #init vars and session
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()
        
        self.logFile.write('Model constructed.')
        print('Complete Model Construction')
    
    def _fullyConnectedLayer(self, x, inputSize, outputSize, use_sig, num):
        W = tf.get_variable("W"+str(num), dtype=tf.float32, shape=[inputSize, outputSize])

        b = tf.get_variable("b"+str(num), shape=[outputSize], dtype=tf.float32, 
                            initializer=tf.zeros_initializer())
            
        layer = tf.matmul(x, W) + b #shape=[batch_size, numClasses]
        
        if use_sig:
            layer = tf.nn.sigmoid(layer)
            
        return layer
        
        
    
    def train(self, trainReader, valReader):
        print('Starting model training')
        #self.add_summary()
        highestScore = 0
        nEpochWithoutImprovement = 0
        
        for nEpoch in range(self.config.nTrainEpochs):
            msg = 'Epoch {} '.format(nEpoch+1)
            print(msg)
            self.logFile.write(msg)

            score = self._run_epoch(trainReader, valReader, nEpoch)
            self.config.lossRate *= self.config.lossRateDecay

            # early stopping and saving best parameters
            if score >= highestScore:
                nEpochWithoutImprovement = 0
                self._save_session()
                highestScore = score
                self.logFile.write('New score')
            else:
                nEpochWithoutImprovement += 1
                if nEpochWithoutImprovement >= self.config.nEpochsWithoutImprov:
                    self.logFile.write('Early stopping at epoch {} with {} epochs\
                                 without improvement'.format(nEpoch+1, nEpochWithoutImprovement))
                    break
    
    def _save_session(self):
        #if not os.path.exists(self.config.dir_model):
        #    os.makedirs(self.config.dir_model)
        self.saver.save(self.sess, self.config.saveModelFile)
    
    def _run_epoch(self, trainReader, valReader, nEpoch):
        '''
        Runs 1 epoch and returns val score
        '''
        # Potentially add progbar here
        batch_size = self.config.batch_size
        
        for i, (qnAsWordIDsBatch, seqLens, img_vecs, labels) in enumerate(
            trainReader.getNextBatch(batch_size)):
            
            feed = {
                self.word_ids : qnAsWordIDsBatch,
                self.sequence_lengths : seqLens,
                self.img_vecs : img_vecs,
                self.labels : labels,
                self.lr : self.config.lossRate,
                self.dropout : self.config.dropoutVal
            }
            _, _ = self.sess.run(
                [self.train_op, self.loss], feed_dict=feed)
            
            if (i%1000==0):
                #every 2000 batches
                feed[self.dropout] = 1.0
                trainAcc = self.sess.run(self.accuracy, feed_dict=feed)
                valAcc = self.runVal(valReader)
                print('Val acc = {}'.format(valAcc))
                resMsg = 'Epoch {0}, batch {1}: val Score={2:>6.1%}, trainAcc={3:>6.1%}\n'.format(
                    nEpoch, i, valAcc, trainAcc)
                self.logFile.write(resMsg)
                print(resMsg)
            
        epochScore = self.runVal(valReader)
        epMsg = 'Epoch {0}: val Score={1:>6.1%}\n'.format(
                    nEpoch, epochScore)
        print(epMsg)
        self.logFile.write(epMsg)
        return epochScore
    
    def runVal(self, valReader):
        """Evaluates performance on test set
        Args:
            test: dataset that yields tuple of (sentences, tags)
        Returns:
            metrics: (dict) metrics["acc"] = 98.4, ...
        """
        accuracies = []
        for qnAsWordIDsBatch, seqLens, img_vecs, labels in valReader.getNextBatch(
            self.config.batch_size):
            feed = {
                self.word_ids : qnAsWordIDsBatch,
                self.sequence_lengths : seqLens,
                self.img_vecs : img_vecs,
                self.labels : labels,
                self.dropout : 1.0
            }
            labels_pred = self.sess.run(self.labels_pred, feed_dict=feed)
            
            for lab, labPreds in zip(labels, labels_pred):
                accuracies.append(lab==labPreds)
        
        valAcc = np.mean(accuracies)
        return valAcc
    
    def loadTrainedModel(self):
        self.sess = tf.Session()
        self.saver = saver = tf.train.import_meta_graph('LSTMIMG-proto.meta')
        saver.restore(self.sess, tf.train.latest_checkpoint('/media/jwong/Transcend/VQADataset/DummySets/'))
        
        graph = tf.get_default_graph()
        self.labels_pred = graph.get_tensor_by_name('labels_pred:0')
        self.accuracy = graph.get_tensor_by_name('accuracy:0')
        self.word_ids = graph.get_tensor_by_name('word_ids:0')
        self.img_vecs = graph.get_tensor_by_name('img_vecs:0')
        self.sequence_lengths = graph.get_tensor_by_name('sequence_lengths:0')
        self.labels = graph.get_tensor_by_name('labels:0')
        
        self.saver = tf.train.Saver()
        
    def destruct(self):
        self.logFile.close()
        
    