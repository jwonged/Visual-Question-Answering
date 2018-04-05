'''
Created on 7 Feb 2018

@author: jwong
'''
import json
import os
import pickle
import time
import os

from model_utils import getPretrainedw2v
import numpy as np
import tensorflow as tf 
from Base_CNNModel import BaseModel

class LSTMCNNModel(BaseModel):
    '''
    VQA Model implementing an E2E CNN with LSTM
    '''
    def __init__(self, config):
        super(LSTMCNNModel, self).__init__(config)
    
    def _addPlaceholders(self):
        # add network placeholders
        # shape = (batch size, max length of sentence in batch)
        self.word_ids = tf.placeholder(tf.int32, shape=[None, None], name="word_ids")
        
        # shape = (batch size, image height, width, channels)
        self.img_vecs = tf.placeholder(tf.float32, 
                                       shape=[None, 224, 224, 3], 
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
        
        return tf.nn.dropout(self.word_embeddings, self.dropout)
    
    def _addLSTM(self, lstmInput):
        #LSTM part
        with tf.variable_scope("lstm"):
            if self.config.LSTMType == 'bi':
                print('Using bi-LSTM')
                cell_fw = tf.contrib.rnn.LSTMCell(self.config.LSTM_num_units)
                cell_bw = tf.contrib.rnn.LSTMCell(self.config.LSTM_num_units)
                
                #Out [batch_size, max_time, cell_output_size] output, outputState.c and .h
                (_, _), (fw_state, bw_state) = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw, cell_bw, 
                    lstmInput, 
                    sequence_length=self.sequence_lengths, dtype=tf.float32)
                print('Shape of fw_state.h: {}'.format(fw_state.h.get_shape()))
                
                #[batch_size, max_time, cell.output_size]
                lstmOutput = tf.concat([fw_state.h, bw_state.h], axis=-1)
                
            else:
                print('Using Uni-LSTM')
                lstm_cell = tf.contrib.rnn.LSTMCell(self.config.LSTM_num_units)
                _, lstmOutState = tf.nn.dynamic_rnn(cell=lstm_cell, 
                                                  inputs=self.word_embeddings, 
                                                  sequence_length=self.sequence_lengths, 
                                                  initial_state=None, 
                                                  dtype=tf.float32)
                lstmOutput = lstmOutState.h #output state 512
        
        lstmOutput = tf.nn.dropout(lstmOutput, self.dropout)
        return lstmOutput
    
    def _addCNNs(self, image):
        #4 conv layers
        with tf.variable_scope("CNNs"):
            
            conv1 = tf.layers.conv2d(inputs=image, 
                                     filters=64,
                                     kernel_size=[3, 3],
                                     padding="same",
                                     activation=tf.nn.relu)
            
            pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
            
            conv2 = tf.layers.conv2d(inputs=pool1, 
                                     filters=64,
                                     kernel_size=[3, 3],
                                     padding="same",
                                     activation=tf.nn.relu)
            
            pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
            
            conv3 = tf.layers.conv2d(inputs=pool2, 
                                     filters=128,
                                     kernel_size=[3, 3],
                                     padding="same",
                                     activation=tf.nn.relu)
            
            pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=2)
            
            conv4 = tf.layers.conv2d(inputs=pool3, 
                                     filters=128,
                                     kernel_size=[3, 3],
                                     padding="same",
                                     activation=tf.nn.relu)
            
            pool4 = tf.layers.max_pooling2d(inputs=conv4, pool_size=[4, 4], strides=4)
            
            print('pool6 shape: {}'.format(tf.shape(pool4)))
            print('pool6 shape: {}'.format(pool4.get_shape()))
            
            #[bx28x28x512]; [bx7x7x256]
            flattenedImgVec = tf.reshape(pool4, [-1, pool4.get_shape()[1:4].num_elements()])
            print('flattened shape: {}'.format(tf.shape(flattenedImgVec)))
            print('flattened shape: {}'.format(flattenedImgVec.get_shape()))
            #401,408 --> 4096; 12544 --> 4096
            reduced_imgVecs = tf.layers.dense(inputs=flattenedImgVec,
                                       units=4096,
                                       activation=tf.tanh,
                                       kernel_initializer=tf.contrib.layers.xavier_initializer())
            return reduced_imgVecs
    
    def construct(self):
        self._addPlaceholders()
        
        self.lstmInput = self._addEmbeddings()
        
        self.lstmOutput = self._addLSTM(self.lstmInput) #[batch_size, max_time, 1024]
        
        self.imgFeatures = self._addCNNs(self.img_vecs) 
        print('img_vecs shape: {}'.format(tf.shape(self.imgFeatures)))
        print('img_vecs shape: {}'.format(self.imgFeatures.get_shape()))
            
        #Combine modes
        if self.config.elMult:
            print('Using pointwise mult')
            #4096 --> 1024
            reduced_imgVecs = tf.layers.dense(inputs=self.imgFeatures,
                                       units=self.lstmOutput.get_shape()[-1],
                                       activation=tf.tanh,
                                       kernel_initializer=tf.contrib.layers.xavier_initializer())
            
            print('reduced_imgVecs shape: {}'.format(tf.shape(reduced_imgVecs)))
            print('reduced_imgVecs shape: {}'.format(reduced_imgVecs.get_shape()))
            
            #dropout after img mapping layer
            reduced_imgVecs = tf.nn.dropout(reduced_imgVecs, self.dropout)
                
            self.multimodalOutput = tf.multiply(self.lstmOutput, reduced_imgVecs) #size=512
        else: #using concat
            print('Using concat')
            self.multimodalOutput = tf.concat([self.lstmOutput, self.imgFeatures ], axis=-1)
    
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
        
        self.predProbs = tf.nn.softmax(y, name='softmax')
        self.topK = tf.nn.top_k(self.predProbs, k=5, name='topK')
        
        #define losses
        crossEntropyLoss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=y, labels=self.labels)
        self.loss = tf.reduce_mean(crossEntropyLoss)

        # Add loss to tensorboard
        tf.summary.scalar("loss", self.loss)
        
        self._addOptimizer()
        
        #init vars and session
        self._initSession()
    
    def _addQuestionAttention(self, lstmOutput):
        #########Question Attention##########
        with tf.variable_scope("qn_attention"):
            #qnAtt_f output: [b x seqLen x 1024]
            qnAtt_f =  tf.layers.dense(lstmOutput, units=lstmOutput.get_shape()[-1],
                                activation=tf.tanh,
                                kernel_initializer=tf.contrib.layers.xavier_initializer()) 
            print('qnAtt_f shape: {}'.format(qnAtt_f.get_shape()))
            print('qnAtt_f shape: {}'.format(tf.shape(qnAtt_f)))
            qnAtt_flat = tf.reshape(qnAtt_f, shape=[-1, qnAtt_f.get_shape()[-1]]) #[b*seqlen, 1024]
            qnAtt_beta = tf.get_variable("beta", shape=[qnAtt_f.get_shape()[-1], 1], dtype=tf.float32)
            
            qnAtt_flatWeights = tf.matmul(qnAtt_flat, qnAtt_beta) #[b*seqLen, 1]
            qnAtt_regionWeights = tf.reshape(
                qnAtt_flatWeights, shape=[-1, tf.shape(lstmOutput)[1]])
            #[b, seqLen(==nRegions)]
            
            self.qnAtt_alpha = tf.nn.softmax(qnAtt_regionWeights, name = 'qn_alpha')
            qnAtt_alpha = tf.expand_dims(self.qnAtt_alpha, axis=-1) #[b, seqLen, 1]
            
            qnContext = tf.reduce_sum(tf.multiply(qnAtt_alpha, lstmOutput), axis=1)
        
        return qnContext
    
    def _addImageAttention(self, qnContext, flattenedImageVecs):
        #########Image Attention layer##########
        with tf.variable_scope("image_attention"):
            qnContext_in = tf.layers.dense(inputs=qnContext,
                                           units=qnContext.get_shape()[-1],
                                           activation=tf.tanh,
                                           kernel_initializer=tf.contrib.layers.xavier_initializer())
            #[bx1024]
            
            #duplicate qn vec to combine with each region to get [v_i, q]
            qnAtt_in = tf.expand_dims(qnContext_in, axis=1)
            qnAtt_in = tf.tile(qnAtt_in, [1,tf.shape(flattenedImageVecs)[1],1]) 
            print('Shape of qnAatt_in : {}'.format(qnAtt_in.get_shape()))
            att_in = tf.concat([flattenedImageVecs, qnAtt_in], axis=-1) #[bx196x1536]
            print('Shape of attention input : {}'.format(att_in.get_shape()))
            
            #compute attention weights
            print('att_in shape: {}'.format(att_in.get_shape()))
            #beta * tanh(wx + b) -- get a scalar val for each region
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
            imgContext = tf.reduce_sum(tf.multiply(alpha, flattenedImageVecs), axis=1) 
            
        return imgContext
    
    def loadTrainedModel(self, restoreModel, restoreModelPath):
        graph = super(LSTMCNNModel, self).loadTrainedModel(restoreModel, restoreModelPath)
        #self.img_alpha = graph.get_tensor_by_name('image_attention/alpha:0')
        #self.qn_alpha = graph.get_tensor_by_name('qn_attention/qn_alpha:0')
    