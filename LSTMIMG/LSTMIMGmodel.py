'''
Created on 15 Jan 2018

@author: jwong
'''

import json
import numpy as np
import tensorflow as tf 
import numpy as np
import pickle

from HelperFunctions import getPretrainedw2v

class LSTMIMGmodel(object):
    '''
    Uses Bi-LSTM to a fully connected layer
    '''

    def __init__(self, config):
        self.config = config
        self.logFile = open(config.logFile, 'w')
        self.logFile.write('Initializing LSTMIMG')
        self.sess   = None
        self.saver  = None

    def construct(self):
        #add placeholders
        self.logFile.write('Constructing model...')
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
            
            #In [batch_size, max_time, ...]
            #fw, bw, inputs, seq len
            #Out [batch_size, max_time, cell_output_size] output, outputState
            (_, _), (fw_state, bw_state) = tf.nn.bidirectional_dynamic_rnn(
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
            LSTMOutputSize = 2*LSTM_num_units + self.config.imgVecSize
        
        #fully connected layer
        with tf.variable_scope("proj"):
            W = tf.get_variable("W", dtype=tf.float32,
                    shape=[LSTMOutputSize, self.config.nOutClasses])

            b = tf.get_variable("b", shape=[self.config.nOutClasses],
                    dtype=tf.float32, initializer=tf.zeros_initializer())
            
            y = tf.matmul(self.LSTMOutput, W) + b
        
        #predict & get accuracy
        self.labels_pred = tf.cast(tf.argmax(y, axis=1), tf.int32)
        is_correct_prediction = tf.equal(self.labels_pred, self.labels)
        self.accuracy = tf.reduce_mean(tf.cast(is_correct_prediction, tf.float32), name='accuracy')
        
        #define losses
        crossEntropyLoss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=y, labels=self.labels)
        #mask = tf.sequence_mask(self.sequence_lengths)
        #losses = tf.boolean_mask(crossEntropyLoss, mask)
        self.loss = tf.reduce_mean(crossEntropyLoss)

        # for tensorboard
        tf.summary.scalar("loss", self.loss)
        
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
        print('Complete')
    
    
    def train(self, trainReader, valReader):
        
        self.add_summary()
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
                self.save_session()
                best_score = score
                self.logFile.write('New score')
            else:
                nEpochWithoutImprovement += 1
                if nEpochWithoutImprovement >= self.config.nEpochsWithoutImprov:
                    self.logFile.write('Early stopping at epoch {} with {} epochs\
                                 without improvement'.format(nEpoch+1, nEpochWithoutImprovement))
                    break
    
    def _run_epoch(self, trainReader, valReader, nEpoch):
        '''
        Runs 1 epoch and returns val score
        '''
        # Potentially add progbar here
        batch_size = self.config.batch_size
        nbatches = (len(train) + batch_size - 1) // batch_size

        # iterate over dataset
        for i, (words, labels) in enumerate(minibatches(train, batch_size)):
            fd, _ = self.get_feed_dict(words, labels, self.config.lr,
                    self.config.dropout)

            _, train_loss, summary = self.sess.run(
                    [self.train_op, self.loss, self.merged], feed_dict=fd)

            prog.update(i + 1, [("train loss", train_loss)])

            # tensorboard
            if i % 10 == 0:
                self.file_writer.add_summary(summary, epoch*nbatches + i)

        metrics = self.run_evaluate(dev)
        msg = " - ".join(["{} {:04.2f}".format(k, v)
                for k, v in metrics.items()])
        self.logger.info(msg)

        return metrics["f1"]
    
    def _get_feed_dict(self, words, labels):
        '''Get dictionary for feeding to LSTM
        Args:
            words: list of list of word IDs (ie list of qns)
        Returns:
            feed: dict {placeholder: value}
        '''
        
        word_ids, sequence_lengths = self.padQuestionIDs(words, 0)
        
        feed = {
            self.word_ids : word_ids,
            self.sequence_length : sequence_lengths,
            self.labels : labels,
            self.lr : self.config.lossRate,
            self.dropout : self.config.dropoutVal
        }
        
        return feed
    
    def padQuestionIDs(self, questions, padding):
        '''
        Pads each list to be same as max length
        args:
            questions: list of list of word IDs (ie a batch of qns)
            padding: symbol to pad with
        '''
        maxLength = max(map(lambda x : len(x), questions))
        #Get length of longest qn
        paddedQuestions, qnLengths = [], []
        for qn in questions:
            qn = list(qn) #ensure list format
            if (len(qn) < maxLength):
                paddedQn = qn + [padding]*(maxLength - len(qn))
                paddedQuestions.append(paddedQn)
            else:
                paddedQuestions.append(qn)
            qnLengths.append(len(qn))
            
        return paddedQuestions, qnLengths
    
    def destruct(self):
        self.logFile.close()
        
    '''
        
        
        
        
    
    
    