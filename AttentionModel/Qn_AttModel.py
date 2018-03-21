'''
Created on 20 Mar 2018

@author: jwong
'''
import csv
import json
import os
import pickle
import time
import os

from model_utils import getPretrainedw2v, generateForSubmission
import numpy as np
import tensorflow as tf 
from Base_AttModel import BaseModel

class QnAttentionModel(object):
    '''
    VQA Model implementing attention over question and images
    '''
    def __init__(self, config):
        super(QnAttentionModel, self).__init__(config)
    
    
    def _addPlaceholders(self):
        # add network placeholders
        self.logFile.writerow(['Constructing model...\n'])
        
        # shape = (batch size, max length of sentence in batch)
        self.word_ids = tf.placeholder(tf.int32, shape=[None, None], name="word_ids")
        
        # shape = (batch size, img tensor dimensions)
        self.img_vecs = tf.placeholder(tf.float32, 
                                       shape=[None, 512, 14, 14], 
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
        with tf.variable_scope("words"):
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
    
    def _addLSTMInput(self):
        #Handle LSTM Input
        print('Constructing imageAfterLSTM model')
        self.LSTMinput = self.word_embeddings
    
    def _addLSTM(self):
        #LSTM part
        with tf.variable_scope("lstm"):
            if self.config.LSTMType == 'bi':
                print('Using bi-LSTM')
                cell_fw = tf.contrib.rnn.LSTMCell(self.config.LSTM_num_units)
                cell_bw = tf.contrib.rnn.LSTMCell(self.config.LSTM_num_units)
                
                #Out [batch_size, max_time, cell_output_size] output, outputState.c and .h
                (output_fw, output_bw), (fw_state, bw_state) = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw, cell_bw, 
                    self.LSTMinput, 
                    sequence_length=self.sequence_lengths, dtype=tf.float32)
                #print('Shape of state.c: {}'.format(fw_state.c.get_shape()))
                print('Shape of output_fw: {}'.format(output_fw.get_shape()))
                print('Shape of output_fw: {}'.format(tf.shape(output_fw)))
                
                #[batch_size, max_time, cell.output_size]
                lstmOutput = tf.concat([output_fw, output_bw], axis=-1)
                
                '''
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
                
                #potentially add dropout here
                '''
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
                lstmOutput = lstmOutState.c #output state 512
                #lstmOutput =  tf.concat([lstmOutState.c, lstmOutState.h], axis=-1) #1024
        
        self.lstmOutput = tf.nn.dropout(lstmOutput, self.dropout)
        
    
    def construct(self):
        self._addPlaceholders()
        
        self._addEmbeddings()
        
        self._addLSTMInput() #[batch_size, max_time, 1024]
        
        #########Question Attention##########
        with tf.variable_scope("qn_attention"):
            #qnAtt_f output: [b x seqLen x 512]
            qnAtt_f =  tf.layers.dense(self.lstmOutput, units=self.lstmOutput.get_shape()[-1],
                                activation=tf.tanh,
                                kernel_initializer=tf.contrib.layers.xavier_initializer()) 
            print('qnAtt_f shape: {}'.format(qnAtt_f.get_shape()))
            print('qnAtt_f shape: {}'.format(tf.shape(qnAtt_f)))
            qnAtt_flat = tf.reshape(qnAtt_f, shape=[-1, qnAtt_f.get_shape()[-1]]) #[b*seqlen, 512]
            qnAtt_beta = tf.get_variable("beta", shape=[qnAtt_f.get_shape()[-1], 1], dtype=tf.float32)
            
            qnAtt_flatWeights = tf.matmul(qnAtt_flat, qnAtt_beta) #[b*seqLen, 1]
            qnAtt_regionWeights = tf.reshape(
                qnAtt_flatWeights, shape=[-1, self.lstmOutput.get_shape()[1]])
            #[b, seqLen(==nRegions)]
            
            self.qnAtt_alpha = tf.nn.softmax(qnAtt_regionWeights, name = 'qn_alpha')
            qnAtt_alpha = tf.expand_dims(self.qnAtt_alpha, axis=-1) #[b, seqLen, 1]
            self.qnContext = tf.reduce_sum(tf.multiply(qnAtt_alpha, self.lstmOutput), axis=1)
            #[b, 1024]
            
        self.batch_size = tf.shape(self.img_vecs)[0]
        print('Batch size = {}'.format(self.batch_size))
        
        #reshape image features [bx512x14x14] --> [bx196x512]
        transposedImgVec = tf.transpose(self.img_vecs, perm=[0,3,2,1]) #bx14x14x512
        print('transposedImgVec = {}'.format(transposedImgVec.get_shape()))
        self.flattenedImgVecs = tf.reshape(transposedImgVec, [self.batch_size, 196, 512])
         
        self._addLSTM()
        
        #########Image Attention layer##########
        with tf.variable_scope("image_attention"):
            qnContext_in = tf.layers.dense(inputs=self.qnContext,
                                           units=self.qnContext.get_shape()[-1],
                                           activation=tf.tanh,
                                           kernel_initializer=tf.contrib.layers.xavier_initializer())
            #[bx1024]
            
            #duplicate qn vec to combine with each region to get [v_i, q]
            qnAtt_in = tf.expand_dims(qnContext_in, axis=1)
            qnAtt_in = tf.tile(qnAtt_in, [1,tf.shape(self.flattenedImgVecs)[1],1]) 
            print('Shape of attention input : {}'.format(tf.shape(qnAtt_in)))
            att_in = tf.concat([self.flattenedImgVecs, qnAtt_in], axis=-1) #[bx196x1536]
            print('Shape of attention input : {}'.format(att_in.get_shape()))
            
            #compute attention weights
            
            #beta * tanh(wx + b) -- get a scalar val for each region
            print('att_in shape: {}'.format(att_in.get_shape()))
            
            att_f = tf.layers.dense(att_in, units=att_in.get_shape()[-1],
                                activation=tf.tanh,
                                kernel_initializer=tf.contrib.layers.xavier_initializer()) #1536
            print('att_f = {}'.format(att_f.get_shape()))
            print('att_f = {}'.format(tf.shape(att_f)))
            beta_w = tf.get_variable("beta", shape=[att_f.get_shape()[-1], 1], dtype=tf.float32) #1536,1
            att_flat = tf.reshape(att_f, shape=[-1, att_f.get_shape()[-1]]) #[b*196, 1536]
            att_flatWeights = tf.matmul(att_flat, beta_w) #get scalar for each batch, region [b*196]
            print('att_flatWeights = {}'.format(att_flatWeights.get_shape()))
            att_regionWeights = tf.reshape(att_flatWeights, shape=[-1, 196])  #[b, 196]
            print('Region weights = {}'.format(att_regionWeights.get_shape()))
            
            #compute context: c = sum alpha * img
            self.alpha = tf.nn.softmax(att_regionWeights, name='alpha') # [b,196]
            alpha = tf.expand_dims(self.alpha, axis=-1)
            
            #broadcast; output shape=[bx1024 or bx1536]
            self.imgContext = tf.reduce_sum(tf.multiply(alpha, self.flattenedImgVecs), axis=1) 
            
        #Combine modes
        if self.config.elMult:
            print('Using pointwise mult')
            #1024 --> 512 or 1536 --> 1024
            attended_img_vecs = tf.layers.dense(inputs=self.imgContext,
                                       units=self.qnContext.get_shape()[-1],
                                       activation=tf.tanh,
                                       kernel_initializer=tf.contrib.layers.xavier_initializer())
            #dropout after img mapping layer
            attended_img_vecs = tf.nn.dropout(attended_img_vecs, self.dropout)
                
            self.multimodalOutput = tf.multiply(self.qnContext, attended_img_vecs) #size=512
        else: #using concat
            print('Using concat')
            self.multimodalOutput = tf.concat([self.qnContext, attended_img_vecs], axis=-1)
    
        #fully connected layer
        with tf.variable_scope("proj"):
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
        
        #define losses
        crossEntropyLoss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=y, labels=self.labels)
        self.loss = tf.reduce_mean(crossEntropyLoss)

        # Add loss to tensorboard
        tf.summary.scalar("loss", self.loss)
        
        self._addOptimizer()
        
        #init vars and session
        self._initSession()
        
    
    def loadTrainedModel(self, restoreModel, restoreModelPath):
        graph = super(QnAttentionModel, self).loadTrainedModel(restoreModel, restoreModelPath)
        self.img_alpha = graph.get_tensor_by_name('image_attention/alpha:0')
        self.qn_alpha = graph.get_tensor_by_name('qn_attention/qn_alpha:0')
        