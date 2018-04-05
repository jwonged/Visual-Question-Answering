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
import csv

from Base_Model import BaseModel
from model_utils import getPretrainedw2v

class LSTMIMGmodel(BaseModel):
    '''
    Uses Bi-LSTM to a fully connected layer
    '''
    def __init__(self, config):
        super(LSTMIMGmodel, self).__init__(config)
        
    def comment(self):
        return 'Standard LSTMIMG model'
    
    def _addPlaceholders(self):
        #add network placeholders
        
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
    
    def _addEmbeddings(self):
        #add word embeddings
        with tf.variable_scope("Word_embeddings"):
            if self.config.usePretrainedEmbeddings:
                print('Using pretrained w2v embeddings')
                pretrainedEmbeddings = getPretrainedw2v(self.config.shortenedEmbeddingsWithUNKFile)
                wordEmbedsVar = tf.Variable(pretrainedEmbeddings,
                        name="wordEmbedsVar",
                        dtype=tf.float32,
                        trainable=self.config.trainEmbeddings)
            else:
                print('Using untrained embeddings')
                wordEmbedsVar = tf.get_variable(
                        name='_word_embeddings',
                        shape=[self.config.vocabSize, self.config.wordVecSize], 
                        dtype=tf.float32)
                
        #embedding matrix, word_ids
        self.word_embeddings = tf.nn.embedding_lookup(wordEmbedsVar,
                self.word_ids, name="word_embeddings")
        
        self.word_embeddings = tf.nn.dropout(self.word_embeddings, self.dropout)
        
    def construct(self):
        self._addPlaceholders()
        
        self._addEmbeddings()
        
        #Handle input according to model structure
        if self.config.modelStruct == 'imagePerWord':
            print('Constructing imagePerWord model')
            self.LSTMinput = tf.concat([self.word_embeddings, self.img_vecs])
            
        elif self.config.modelStruct == 'imageAsFirstWord':
            with tf.variable_scope('Image_mapping'):
                print('Constructing imagePerWord model')
                #map image 1024 --> 512 --> 300
                imgMappingLayer1 = tf.layers.dense(inputs=self.img_vecs,
                                               units=self.config.imgVecSize/2, 
                                               activation=tf.nn.relu,
                                               kernel_initializer=tf.contrib.layers.xavier_initializer())
                imgMappingLayer2 = tf.layers.dense(inputs=imgMappingLayer1,
                                               units=self.config.wordVecSize,
                                               activation=tf.nn.relu,
                                               kernel_initializer=tf.contrib.layers.xavier_initializer())
                
                #reshape to allow concat with word embeddings
                imgMappingLayer2 = tf.reshape(
                    imgMappingLayer2, [self.config.batch_size, 1, self.config.wordVecSize])
                print('Shape of img map layer 2: {}'.format(imgMappingLayer2.get_shape()))
                
                #add img embedding as first word to lstm input
                self.LSTMinput = tf.concat([imgMappingLayer2, self.word_embeddings], axis=1)
                print('Shape of LSTM input: {}'.format(self.LSTMinput.get_shape()))
                
                #add 1 to all sequence lengths to account for extra img word
                self.sequence_lengths = tf.add(
                    self.sequence_lengths, tf.ones(tf.shape(self.sequence_lengths), dtype=tf.int32))
        
        else:
            print('Constructing imageAfterLSTM model')
            self.LSTMinput = self.word_embeddings
            
            
        with tf.variable_scope("lstm"):
            if self.config.LSTMType == 'bi':
                print('Using bi-LSTM')
                cell_fw = tf.contrib.rnn.LSTMCell(self.config.LSTM_num_units)
                cell_bw = tf.contrib.rnn.LSTMCell(self.config.LSTM_num_units)
                
                #Out [batch_size, max_time, cell_output_size] output, outputState
                (_, _), (fw_state, bw_state) = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw, cell_bw, 
                    self.word_embeddings, 
                    sequence_length=self.sequence_lengths, dtype=tf.float32)
                print('Shape of state.c: {}'.format(fw_state.c.get_shape())) #[?, 300]
                
                #lstmOutput shape = LSTM_num_units * 4
                fw_out = tf.concat([fw_state.c, fw_state.h], axis=-1)
                bw_out = tf.concat([bw_state.c, bw_state.h], axis=-1)
                
                lstmOutput = tf.concat([fw_out, bw_out], axis=-1)
                print('Shape of LSTM output after concat: {}'.format(lstmOutput.get_shape()))
                
                #lstm output 2048 --> 1024
                lstmOutput = tf.layers.dense(inputs=lstmOutput,
                                           units=self.config.fclayerAfterLSTM,
                                           activation=tf.tanh,
                                           kernel_initializer=tf.contrib.layers.xavier_initializer())
                
                
            else:
                print('Using Uni-LSTM')
                #rnn_layers = [tf.nn.rnn_cell.LSTMCell(size) for size in self.config.LSTMCellSizes]
                #multi_rnn_cell = tf.nn.rnn_cell.MultiRNNCell(rnn_layers)
                lstm_cell = tf.contrib.rnn.LSTMCell(self.config.LSTM_num_units)
                _, lstmOutState = tf.nn.dynamic_rnn(cell=lstm_cell, 
                                                  inputs=self.word_embeddings, 
                                                  sequence_length=self.sequence_lengths, 
                                                  initial_state=None, 
                                                  dtype=tf.float32)
                lstmOutput =  tf.concat([lstmOutState.c, lstmOutState.h], axis=-1)
                
        #dropout after LSTM
        lstmOutput = tf.nn.dropout(lstmOutput, self.dropout)
            
        #Handle output according to model structure
        with tf.variable_scope('Combine_modes'):
            if self.config.modelStruct == 'imagePerWord':
                self.multimodalOutput = lstmOutput 
            elif self.config.modelStruct == 'imageAsFirstWord':
                self.multimodalOutput = lstmOutput
            else: #imageAfterLSTM
                if self.config.elMult:
                    print('Using pointwise mult')
                    #img vecs 4096 --> 2048 (for vgg)
                    img_vecs = tf.layers.dense(inputs=self.img_vecs,
                                               units=self.config.fclayerAfterLSTM,
                                               activation=tf.tanh,
                                               kernel_initializer=tf.contrib.layers.xavier_initializer())
                    #dropout after img mapping layer
                    img_vecs = tf.nn.dropout(img_vecs, self.dropout)
                        
                    self.multimodalOutput = tf.multiply(lstmOutput, img_vecs) #size=1024
                else: #using concat
                    print('Using concat')
                    self.multimodalOutput = tf.concat([lstmOutput, self.img_vecs], axis=-1)
        
        #fully connected layer
        with tf.variable_scope("proj"):
            #hidden_layer1 = tf.layers.dense(inputs=self.multimodalOutput,
            #                               units=LSTMOutputSize/2,
            #                               activation=tf.tanh,
            #                               kernel_initializer=tf.contrib.layers.xavier_initializer())
            hidden_layer2 = tf.layers.dense(inputs=self.multimodalOutput,
                                           units=1000,
                                           activation=tf.tanh,
                                           kernel_initializer=tf.contrib.layers.xavier_initializer())
            y = tf.layers.dense(inputs=hidden_layer2,
                                           units=self.config.nOutClasses,
                                           activation=None,
                                           kernel_initializer=tf.contrib.layers.xavier_initializer())
            print('Shape of y: {}'.format(y.get_shape()))
            
        #predict & get accuracy
        self.labels_pred = tf.cast(tf.argmax(tf.nn.softmax(y), axis=1), tf.int32, name='labels_pred')
        
        is_correct_prediction = tf.equal(self.labels_pred, self.labels)
        self.accuracy = tf.reduce_mean(tf.cast(is_correct_prediction, tf.float32), name='accuracy')
        
        self.predProbs = tf.nn.softmax(y, name='softmax')
        self.predscore = tf.reduce_max(self.predProbs, axis=1, name='predscore')
        self.topK = tf.nn.top_k(self.predProbs, name='topK')
        
        #define losses
        with tf.variable_scope('loss'):
            crossEntropyLoss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                        logits=y, labels=self.labels)
            self.loss = tf.reduce_mean(crossEntropyLoss)

        #Add loss to tensorboard
        tf.summary.scalar("loss", self.loss)
        
        self._addOptimizer()
        
        #init vars and session
        self._initSession()
        
    def loadTrainedModel(self, restoreModel, restoreModelPath):
        print('Restoring model from: {}'.format(restoreModel))
        self.sess = tf.Session()
        self.saver = saver = tf.train.import_meta_graph(restoreModel)
        saver.restore(self.sess, tf.train.latest_checkpoint(restoreModelPath))
        tf.reset_default_graph()
        
        graph = tf.get_default_graph()
        self.labels_pred = graph.get_tensor_by_name('labels_pred:0')
        self.accuracy = graph.get_tensor_by_name('accuracy:0')
        self.word_ids = graph.get_tensor_by_name('word_ids:0')
        self.img_vecs = graph.get_tensor_by_name('img_vecs:0')
        self.sequence_lengths = graph.get_tensor_by_name('sequence_lengths:0')
        self.labels = graph.get_tensor_by_name('labels:0')
        self.dropout = graph.get_tensor_by_name('dropout:0')
        self.topK = graph.get_tensor_by_name('topK:0')
        
        self.saver = tf.train.Saver()
        
    def destruct(self):
        pass
        
    