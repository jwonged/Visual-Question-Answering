'''
Created on 15 Jan 2018

@author: jwong
'''

import csv
import json
import os

from model.Base_AttModel import BaseModel
from utils.model_utils import getPretrainedw2v
import tensorflow as tf 

class ImageAttentionModel(BaseModel):
    '''
    VQA Model implementing attention over images
    '''
    def __init__(self, config):
        super(ImageAttentionModel, self).__init__(config)
    
    def _addPlaceholders(self):
        # add network placeholder
        
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
                
                #Out [batch_size, max_time, cell_output_size] output, outputState
                (_, _), (fw_state, bw_state) = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw, cell_bw, 
                    self.LSTMinput, 
                    sequence_length=self.sequence_lengths, dtype=tf.float32)
                print('Shape of state.c: {}'.format(fw_state.c.get_shape()))
                
                #lstmOutput shape = LSTM_num_units * 4
                #fw_out = tf.concat([fw_state.c, fw_state.h], axis=-1)
                #bw_out = tf.concat([bw_state.c, bw_state.h], axis=-1)
                
                #lstmOutput = tf.concat([fw_out, bw_out], axis=-1)
                lstmOutput = tf.concat([fw_state.h, bw_state.h], axis=-1) #1024
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
                lstmOutput = lstmOutState.c #output state 512
                #lstmOutput =  tf.concat([lstmOutState.c, lstmOutState.h], axis=-1) #1024
        
        self.lstmOutput = tf.nn.dropout(lstmOutput, self.dropout)
    
    
    def _multimodalAttentionShared(self, imgContext, qnContext):
        """
        args:
            imgContext: [b, 512]
            qnContext: [b, 1024]
        """
        print('Using Crossmodal Attention')
        self.im_con = imgContext
        self.qn_con = qnContext
        
        #map qn down to 512
        qnContext = tf.layers.dense(inputs=qnContext,
                                       units=imgContext.get_shape()[-1],
                                       activation=tf.tanh,
                                       kernel_initializer=tf.contrib.layers.xavier_initializer()) #[b,512]
        
        #combine for result
        att_im = tf.expand_dims(imgContext, axis=1) #[b,1,512]
        att_qn = tf.expand_dims(qnContext, axis=1) #[b,1,512]
        mm_start = tf.concat([att_im, att_qn], axis=1) #[b,2,512]
        
        #use the same attention weights for across modes
        combinedInfo = tf.concat([imgContext, qnContext], axis=-1)
        
        att_im = tf.layers.dense(inputs=combinedInfo,
                                       units=combinedInfo.get_shape()[-1],
                                       activation=tf.tanh,
                                       kernel_initializer=tf.contrib.layers.xavier_initializer())#[b,1024]
        att_qn = tf.layers.dense(inputs=combinedInfo,
                                      units=combinedInfo.get_shape()[-1],
                                      activation=tf.tanh,
                                      kernel_initializer=tf.contrib.layers.xavier_initializer()) #[b,1024]
        
        att_im = tf.expand_dims(att_im, axis=1) #[b,1,1024]
        att_qn = tf.expand_dims(att_qn, axis=1) #[b,1,1024]
        mm_in = tf.concat([att_im, att_qn], axis=1) #[b,2,1024]
        
        #beta * tanh(Wx+b)
        #mm_a = tf.layers.dense(inputs=mm_in,
        #                               units=mm_in.get_shape()[-1],
        #                               activation=tf.tanh,
        #                               kernel_initializer=tf.contrib.layers.xavier_initializer())
        mm_beta_w = tf.get_variable("beta", shape=[mm_in.get_shape()[-1], 1], dtype=tf.float32) #512,1
        mm_flat = tf.reshape(mm_in, shape=[-1, mm_in.get_shape()[-1]]) #[b*2, 1024]
        mm_att_flatWeights = tf.matmul(mm_flat, mm_beta_w) #get scalar for each batch, region [b*2]
        mm_a_shaped = tf.reshape(mm_att_flatWeights, shape=[-1, 2])  #[b, 2]
        
        unnorm_alpha = tf.nn.sigmoid(mm_a_shaped) #[b, 2]
        norm_denominator = tf.expand_dims(
            tf.reduce_sum(unnorm_alpha, axis=-1), axis=-1) #[b, 1]
        self.mmAlpha = tf.div(unnorm_alpha, norm_denominator, name='mmAlpha') #[b, 2]
        
        self.mmAlpha_im = self.mmAlpha
        self.mmAlpha_qn = self.mmAlpha
        
        self.alpha_mm = tf.expand_dims(self.mmAlpha, axis=-1)  #[b,2,1]
        mmContext = tf.reduce_sum(tf.multiply(self.alpha_mm, mm_start),  axis=1) #[b,512]
        
        return mmContext
    
    def construct(self):
        self._addPlaceholders()
        
        self._addEmbeddings()
        
        self._addLSTMInput()
        
        self.batch_size = tf.shape(self.img_vecs)[0]
        print('Batch size = {}'.format(self.batch_size))
        
        #reshape image features [bx512x14x14] --> [bx196x512]
        transposedImgVec = tf.transpose(self.img_vecs, perm=[0,3,2,1]) #bx14x14x512
        print('transposedImgVec = {}'.format(transposedImgVec.get_shape()))
        self.flattenedImgVecs = tf.reshape(transposedImgVec, [self.batch_size, 196, 512])
        
        #tanh activate image
        #self.flattenedImgVecs= tf.layers.dense(inputs=self.flattenedImgVecs,
        #                               units=self.flattenedImgVecs.get_shape()[-1],
        #                               activation=tf.tanh,
        #                               kernel_initializer=tf.contrib.layers.xavier_initializer())
         
        self._addLSTM()
        
        #########Attention layer##########
        with tf.variable_scope("attention"):
            lstmOutputForIm = tf.layers.dense(inputs=self.lstmOutput,
                                           units=1024,
                                           activation=tf.tanh,
                                           kernel_initializer=tf.contrib.layers.xavier_initializer())
                
            #duplicate qn vec to combine with each region to get [v_i, q]
            qnAtt_in = tf.expand_dims(lstmOutputForIm, axis=1)
            qnAtt_in = tf.tile(qnAtt_in, [1,tf.shape(self.flattenedImgVecs)[1],1]) 
            print('Shape of attention input : {}'.format(tf.shape(qnAtt_in)))
            att_in = tf.concat([self.flattenedImgVecs, qnAtt_in], axis=-1) #[bx196x1536]
            print('Shape of attention input : {}'.format(att_in.get_shape()))
            
            #compute attention weights
            ''''w = tf.get_variable('w', 
                                shape=[att_in.get_shape()[-1], att_in.get_shape()[-1]], 
                                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.get_variable('b', 
                                shape=[att_in.get_shape()[-1]], 
                                initializer=tf.contrib.layers.xavier_initializer())
            print('Shape of attention weight matrix: {}'.format(w.get_shape()))
            print('Shape of attention bias : {}'.format(b.get_shape()))'''
            
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
            if self.config.attentionFunc == 'softmax':
                self.alpha = tf.nn.softmax(att_regionWeights, name='alpha') # [b,196]
            elif self.config.attentionFunc == 'sigmoid':
                print('Using sigmoid attention function')
                unnorm_alpha = tf.nn.sigmoid(att_regionWeights, name='alpha')
                norm_denominator = tf.expand_dims(
                    tf.reduce_sum(unnorm_alpha, axis=-1), axis=-1)
                self.alpha = unnorm_alpha / norm_denominator
            else:
                raise NotImplementedError
            
            alpha = tf.expand_dims(self.alpha, axis=-1)
            
            #broadcast; output shape=[bx1024 or bx1536]
            self.imgContext = tf.reduce_sum(tf.multiply(alpha, self.flattenedImgVecs), axis=1) 
            
        #Handle output according to model structure
        if self.config.mmAtt:
            #self.multimodalOutput = self._multimodalAttention(self.imgContext, self.lstmOutput)
            self.multimodalOutput = self._multimodalAttentionShared(self.imgContext, self.lstmOutput)
        else:
            if self.config.elMult:
                print('Using pointwise mult')
                #1024 --> 512 or 1536 --> 1024
                attended_img_vecs = tf.layers.dense(inputs=self.imgContext,
                                           units=self.lstmOutput.get_shape()[-1],
                                           activation=tf.tanh,
                                           kernel_initializer=tf.contrib.layers.xavier_initializer())
                #dropout after img mapping layer
                attended_img_vecs = tf.nn.dropout(attended_img_vecs, self.dropout)
                    
                self.multimodalOutput = tf.multiply(self.lstmOutput, attended_img_vecs) #size=512
            else: #using concat
                print('Using concat')
                self.multimodalOutput = tf.concat([self.lstmOutput, attended_img_vecs], axis=-1)
        
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
        
        self.predProbs = tf.nn.softmax(y, name='predProb')
        self.topK = tf.nn.top_k(self.predProbs, k=5, name='topK')

        # Add to tensorboard
        tf.summary.scalar("loss", self.loss)
        tf.summary.scalar("accuracy", self.accuracy)
        
        self._addOptimizer()
        
        #init vars and session
        self._initSession()
        
    def loadTrainedModel(self, restoreModel, restoreModelPath):
        graph = super(ImageAttentionModel, self).loadTrainedModel(restoreModel, restoreModelPath)
        self.alpha = graph.get_tensor_by_name('attention/alpha:0')
        self.mmAlpha = graph.get_tensor_by_name('mmAlpha:0')
    
    def solve(self, qn, img_id, processor):
        qnAsWordIDsBatch, seqLens, img_vecs = processor.processInput(qn, img_id)
        feed = {
                self.word_ids : qnAsWordIDsBatch,
                self.sequence_lengths : seqLens,
                self.img_vecs : img_vecs,
                self.dropout : 1.0
        }
        alphas, labels_pred = self.sess.run([self.alpha, self.labels_pred], feed_dict=feed)
        return alphas[0], self.classToAnsMap[labels_pred[0]]
    