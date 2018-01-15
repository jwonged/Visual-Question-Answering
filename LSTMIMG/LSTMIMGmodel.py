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
        self.sess   = None
        self.saver  = None
        self.logFile.write('Initializing LSTMIMG')

    def construct(self):
        #add placeholders
        self.logFile('Constructing model...')
        # shape = (batch size, max length of sentence in batch)
        self.word_ids = tf.placeholder(tf.int32, shape=[None, None], name="word_ids")

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
        word_embeddings = tf.nn.embedding_lookup(wordEmbedsVar,
                self.word_ids, name="word_embeddings")
        
        self.word_embeddings =  tf.nn.dropout(word_embeddings, self.dropout)
        
        #add logits
        LSTM_num_units = 300
        with tf.variable_scope("bi-lstm"):
            cell_fw = tf.contrib.rnn.LSTMCell(LSTM_num_units)
            cell_bw = tf.contrib.rnn.LSTMCell(LSTM_num_units)
            
            #In [batch_size, max_time, ...]
            #fw, bw, inputs, seq len
            #Out [batch_size, max_time, cell_output_size] output, outputState
            (fwOutput, bwOutput), _ = tf.nn.bidirectional_dynamic_rnn(
                cell_fw, cell_bw, 
                self.word_embeddings, 
                sequence_length=self.sequence_lengths, dtype=tf.float32)
            lstmOutput = tf.concat([fwOutput, bwOutput], axis=-1)
            
            nnOutput = tf.nn.dropout(lstmOutput, self.dropout)
        
        #connected layer
        with tf.variable_scope("proj"):
            W = tf.get_variable("W", dtype=tf.float32,
                    shape=[2*LSTM_num_units, self.config.nOutClasses])

            b = tf.get_variable("b", shape=[self.config.nOutClasses],
                    dtype=tf.float32, initializer=tf.zeros_initializer())

            nsteps = tf.shape(nnOutput)[1]
            output = tf.reshape(nnOutput, [-1, 2*LSTM_num_units])
            pred = tf.matmul(output, W) + b
            self.logits = tf.reshape(pred, [-1, nsteps, self.config.nOutClasses])
        
        #predict
        self.labels_pred = tf.cast(tf.argmax(self.logits, axis=-1), tf.int32)
        
        #define losses
        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=self.logits, labels=self.labels)
        mask = tf.sequence_mask(self.sequence_lengths)
        losses = tf.boolean_mask(losses, mask)
        self.loss = tf.reduce_mean(losses)

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
                self.train_op = optimizer.apply_gradients(zip(grads, vs))
            else:
                self.train_op = optimizer.minimize(self.loss)
                
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

            score = self.run_epoch(train, dev, epoch)
            self.config.lossRate *= self.config.lossRateDecay

            # early stopping and saving best parameters
            if score >= best_score:
                nEpochWithoutImprovement = 0
                self.save_session()
                best_score = score
                self.logger.info("- new best score!")
            else:
                nEpochWithoutImprovement += 1
                if nEpochWithoutImprovement >= self.config.nEpochImproveStop:
                    
                    self.logger.info("- early stopping {} epochs without "\
                            "improvement".format(nepoch_no_imprv))
                    break
    
    def run_epoch(self, train, dev, epoch):
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
    
    def _get_feed_dict(self, words, labels=None, lr=None, dropout=None):
        """Given some data, pad it and build a feed dictionary
        Args:
            words: list of sentences. A sentence is a list of ids of a list of
                words. A word is a list of ids
            labels: list of ids
            lr: (float) learning rate
            dropout: (float) keep prob
        Returns:
            dict {placeholder: value}
        """
        # perform padding of the given data
        if self.config.use_chars:
            char_ids, word_ids = zip(*words)
            word_ids, sequence_lengths = pad_sequences(word_ids, 0)
            char_ids, word_lengths = pad_sequences(char_ids, pad_tok=0,
                nlevels=2)
        else:
            word_ids, sequence_lengths = pad_sequences(words, 0)

        # build feed dictionary
        feed = {
            self.word_ids: word_ids,
            self.sequence_lengths: sequence_lengths
        }

        if self.config.use_chars:
            feed[self.char_ids] = char_ids
            feed[self.word_lengths] = word_lengths

        if labels is not None:
            labels, _ = pad_sequences(labels, 0)
            feed[self.labels] = labels

        if lr is not None:
            feed[self.lr] = lr

        if dropout is not None:
            feed[self.dropout] = dropout

        return feed, sequence_lengths
    
    def _pad_sequences(self, sequences, pad_tok, max_length):
        """Args:
            sequences: a generator of list or tuple
            pad_tok: the char to pad with
        Returns:
            a list of list where each sublist has same length
        """
        sequence_padded, sequence_length = [], []
    
        for seq in sequences:
            seq = list(seq)
            seq_ = seq[:max_length] + [pad_tok]*max(max_length - len(seq), 0)
            sequence_padded +=  [seq_]
            sequence_length += [min(len(seq), max_length)]

    return sequence_padded, sequence_length

    def pad_sequences(self, sequences, pad_tok, nlevels=1):
        """
        Args:
            sequences: a generator of list or tuple
            pad_tok: the char to pad with
            nlevels: "depth" of padding, for the case where we have characters ids
        Returns:
            a list of list where each sublist has same length
        """
        if nlevels == 1:
            max_length = max(map(lambda x : len(x), sequences))
            sequence_padded, sequence_length = _pad_sequences(sequences,
                                                pad_tok, max_length)
    
        elif nlevels == 2:
            max_length_word = max([max(map(lambda x: len(x), seq))
                                   for seq in sequences])
            sequence_padded, sequence_length = [], []
            for seq in sequences:
                # all words are same length now
                sp, sl = _pad_sequences(seq, pad_tok, max_length_word)
                sequence_padded += [sp]
                sequence_length += [sl]
    
            max_length_sentence = max(map(lambda x : len(x), sequences))
            sequence_padded, _ = _pad_sequences(sequence_padded,
                    [pad_tok]*max_length_word, max_length_sentence)
            sequence_length, _ = _pad_sequences(sequence_length, 0,
                    max_length_sentence)

    return sequence_padded, sequence_length
            
    def destruct(self):
        self.logFile.close()
        
        
        
        
        
        
    
    
    