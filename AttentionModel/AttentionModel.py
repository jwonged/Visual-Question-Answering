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

from model_utils import getPretrainedw2v

class AttentionModel(object):
    '''
    VQA Model implementing attention
    '''

    def __init__(self, config):
        self.config = config
        
        f1 = open(config.logFile, 'wb')
        self.logFile = csv.writer(f1)
        self.logFile.writerow(['Initializing LSTMIMG\n'])
        self.logFile.writerow([self._getDescription(config)])
        
        f2 =  open(config.csvResults , 'wb') 
        self.predFile = csv.writer(f2)
        self.predFile.writerow(
            ['Epoch','Question', 'Prediction', 'Label', 'Pred Class',
             'label class', 'Correct?', 'img id', 'qn_id'])
        
        self.classToAnsMap = config.classToAnsMap
        self.sess   = None
        self.saver  = None
        
    def _getDescription(self, config):
        info = 'model: {}, classes: {}, batchSize: {}, \
            dropout: {}, optimizer: {}, lr: {}, decay: {}, \
             clip: {}, shuffle: {}, trainEmbeddings: {}, LSTM_units: {}, \
             usePretrainedEmbeddings: {}, LSTMType: {}, elMult: {}, imgModel: {}'.format(
                config.modelStruct, config.nOutClasses, config.batch_size,
                config.dropoutVal, config.modelOptimizer, config.lossRate,
                config.lossRateDecay, config.max_gradient_norm, config.shuffle,
                config.trainEmbeddings, config.LSTM_num_units, config.usePretrainedEmbeddings,
                config.LSTMType, config.elMult, config.imgModel)
        return info + 'fc: 2 layers (1000)'
    
    def _addPlaceholders(self):
        #add network placeholders
        self.logFile.writerow(['Constructing model...\n'])
        
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
        
        if self.config.dropoutVal < 1.0:
            self.word_embeddings = tf.nn.dropout(self.word_embeddings, self.dropout)
        
    def construct(self):
        self._addPlaceholders()
        
        self._addEmbeddings()
        
        #Handle input according to model structure
        if self.config.modelStruct == 'imagePerWord':
            print('Constructing imagePerWord model')
            self.LSTMinput = tf.concat([self.word_embeddings, self.img_vecs])
            
        elif self.config.modelStruct == 'imageAsFirstWord':
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
            
            
        #LSTM part
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
        if self.config.dropoutVal < 1.0:
            lstmOutput = tf.nn.dropout(lstmOutput, self.dropout)
            
        #Handle output according to model structure
        if self.config.modelStruct == 'imagePerWord':
            self.LSTMOutput = lstmOutput 
        elif self.config.modelStruct == 'imageAsFirstWord':
            self.LSTMOutput = lstmOutput
        else: #imageAfterLSTM
            if self.config.elMult:
                print('Using pointwise mult')
                #img vecs 4096 --> 2048 (for vgg)
                img_vecs = tf.layers.dense(inputs=self.img_vecs,
                                           units=self.config.fclayerAfterLSTM,
                                           activation=tf.tanh,
                                           kernel_initializer=tf.contrib.layers.xavier_initializer())
                #dropout after img mapping layer
                if self.config.dropoutVal < 1.0:
                    img_vecs = tf.nn.dropout(img_vecs, self.dropout)
                    
                self.LSTMOutput = tf.multiply(lstmOutput, img_vecs) #size=1024
            else: #using concat
                print('Using concat')
                self.LSTMOutput = tf.concat([lstmOutput, self.img_vecs], axis=-1)
        
        #fully connected layer
        with tf.variable_scope("proj"):
            #hidden_layer1 = tf.layers.dense(inputs=self.LSTMOutput,
            #                               units=LSTMOutputSize/2,
            #                               activation=tf.tanh,
            #                               kernel_initializer=tf.contrib.layers.xavier_initializer())
            hidden_layer2 = tf.layers.dense(inputs=self.LSTMOutput,
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

        # for tensorboard
        #tf.summary.scalar("loss", self.loss)
        
        self._addOptimizer()
        
        #init vars and session
        self._initSession()
        
    
    def _addOptimizer(self):
        #train optimizer
        with tf.variable_scope("train_step"):
            if self.config.modelOptimizer == 'adam': 
                print('Using adam optimizer')
                optimizer = tf.train.AdamOptimizer(self.lr)
            elif self.config.modelOptimizer == 'adagrad':
                print('Using adagrad optimizer')
                optimizer = tf.train.AdagradOptimizer(self.lr)
            else:
                print('Using grad desc optimizer')
                optimizer = tf.train.GradientDescentOptimizer(self.lr)
                
            if self.config.max_gradient_norm > 0: # gradient clipping if clip is positive
                grads, vs     = zip(*optimizer.compute_gradients(self.loss))
                grads, gnorm  = tf.clip_by_global_norm(grads, self.config.max_gradient_norm)
                self.train_op = optimizer.apply_gradients(zip(grads, vs), name='trainModel')
            else:
                self.train_op = optimizer.minimize(self.loss, name='trainModel')
    
    def _initSession(self):
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()#max_to_keep=4)
        
        self.logFile.writerow(['Model constructed.'])
        print('Complete Model Construction')
        
    def train(self, trainReader, valReader):
        print('Starting model training')
        self.logFile.writerow([
            'Epoch', 'Val score', 'Train score', 'Train correct', 
            'Train predictions', 'Val correct', 'Val predictions'])
        #self.add_summary()
        highestScore = 0
        nEpochWithoutImprovement = 0
        
        for nEpoch in range(self.config.nTrainEpochs):
            msg = 'Epoch {} \n'.format(nEpoch)
            print(msg)
            #self.logFile.write(msg)

            score = self._run_epoch(trainReader, valReader, nEpoch)
            self.config.lossRate *= self.config.lossRateDecay

            # early stopping and saving best parameters
            if score >= highestScore:
                nEpochWithoutImprovement = 0
                if nEpoch == 0:
                    self.saver.save(self.sess, self.config.saveModelFile, global_step=nEpoch)
                else:
                    self.saver.save(self.sess, self.config.saveModelFile, 
                                    global_step=nEpoch,  write_meta_graph=False)
                #self._save_session()
                highestScore = score
                #self.logFile.writerow('New score\n')
            else:
                nEpochWithoutImprovement += 1
                if nEpochWithoutImprovement >= self.config.nEpochsWithoutImprov:
                    self.logFile.writerow([
                        'Early stopping at epoch {} with {} epochs without improvement'.format(
                            nEpoch+1, nEpochWithoutImprovement)])
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
        correct_predictions, total_predictions = 0., 0.
        
        for i, (qnAsWordIDsBatch, seqLens, img_vecs, labels, _, _, _) in enumerate(
            trainReader.getNextBatch(batch_size)):
            
            feed = {
                self.word_ids : qnAsWordIDsBatch,
                self.sequence_lengths : seqLens,
                self.img_vecs : img_vecs,
                self.labels : labels,
                self.lr : self.config.lossRate,
                self.dropout : self.config.dropoutVal
            }
            _, _, labels_pred = self.sess.run(
                [self.train_op, self.loss, self.labels_pred], feed_dict=feed)
            
            for lab, labPred in zip(labels, labels_pred):
                if lab==labPred:
                    correct_predictions += 1
                total_predictions += 1
                
                #log to csv
                #self.predFile.writerow([qn, self.classToAnsMap[labPred], self.classToAnsMap[lab], labPred, lab, lab==labPred])
                #self.predFile.write('Qn:{}, lab:{}, pred:{}\n'.format(qn, self.classToAnsMap[lab], self.classToAnsMap[labPred]))
                '''
            if (i%4000==0):
                valAcc, valCorrect, valTotalPreds = self.runVal(valReader, nEpoch)
                resMsg = 'Epoch {0}, batch {1}: val Score={2:>6.1%}, trainAcc={3:>6.1%}\n'.format(
                    nEpoch, i, valAcc, correct_predictions/total_predictions if correct_predictions > 0 else 0 )
                self.logFile.write(resMsg)
                print(resMsg)'''
            
        epochScore, valCorrect, valTotalPreds = self.runVal(valReader, nEpoch)
        trainScore = correct_predictions/total_predictions if correct_predictions > 0 else 0
        epMsg = 'Epoch {0}: val Score={1:>6.1%}, train Score={2:>6.1%}, total train predictions={3}\n'.format(
                    nEpoch, epochScore, trainScore, total_predictions)
        print(epMsg)
        self.logFile.writerow([
            nEpoch, epochScore, trainScore, correct_predictions, total_predictions, valCorrect, valTotalPreds])
        return epochScore
    
    def runVal(self, valReader, nEpoch):
        """Evaluates performance on test set
        Args:
            test: dataset that yields tuple of (sentences, tags)
        Returns:
            metrics: (dict) metrics["acc"] = 98.4, ...
        """
        accuracies = []
        correct_predictions, total_predictions = 0., 0.
        for qnAsWordIDsBatch, seqLens, img_vecs, labels, rawQns, img_ids,_ in \
            valReader.getNextBatch(self.config.batch_size):
            feed = {
                self.word_ids : qnAsWordIDsBatch,
                self.sequence_lengths : seqLens,
                self.img_vecs : img_vecs,
                self.labels : labels,
                self.dropout : 1.0
            }
            labels_pred = self.sess.run(self.labels_pred, feed_dict=feed)
            
            for lab, labPred, qn ,img_id in zip(labels, labels_pred, rawQns, img_ids):
                if (lab==labPred):
                    correct_predictions += 1
                total_predictions += 1
                accuracies.append(lab==labPred)
                #self._logToCSV(
                #    nEpoch, qn, 
                #    self.classToAnsMap[labPred], 
                #    self.classToAnsMap[lab], 
                #    labPred, lab, lab==labPred, img_id)
                
        valAcc = np.mean(accuracies)
        return valAcc, correct_predictions, total_predictions
    
    def _logToCSV(self, nEpoch, qn, prediction, label, 
                  predClass, labelClass, correct, img_id):
        self.predFile.writerow(
                    [nEpoch, qn, prediction, label, predClass, 
                     labelClass, correct, img_id])
        
    def loadTrainedModel(self):
        restoreModel = self.config.restoreModel
        print('Restoring model from: {}'.format(restoreModel))
        self.sess = tf.Session()
        self.saver = saver = tf.train.import_meta_graph(restoreModel)
        saver.restore(self.sess, tf.train.latest_checkpoint(self.config.restoreModelPath))
        
        graph = tf.get_default_graph()
        self.labels_pred = graph.get_tensor_by_name('labels_pred:0')
        self.accuracy = graph.get_tensor_by_name('accuracy:0')
        self.word_ids = graph.get_tensor_by_name('word_ids:0')
        self.img_vecs = graph.get_tensor_by_name('img_vecs:0')
        self.sequence_lengths = graph.get_tensor_by_name('sequence_lengths:0')
        self.labels = graph.get_tensor_by_name('labels:0')
        self.dropout = graph.get_tensor_by_name('dropout:0')
        
        self.saver = tf.train.Saver()
        
    def runPredict(self, valReader):
        '''For internal val/test set with labels'''
        accuracies = []
        correct_predictions, total_predictions = 0., 0.
        for qnAsWordIDsBatch, seqLens, img_vecs, labels, rawQns, img_ids, qn_ids \
            in valReader.getNextBatch(self.config.batch_size):
            feed = {
                self.word_ids : qnAsWordIDsBatch,
                self.sequence_lengths : seqLens,
                self.img_vecs : img_vecs,
                self.dropout : 1.0
            }
            labels_pred = self.sess.run(self.labels_pred, feed_dict=feed)
            
            for lab, labPred, qn, img_id, qn_id in zip(
                labels, labels_pred, rawQns, img_ids, qn_ids):
                if (lab==labPred):
                    correct_predictions += 1
                total_predictions += 1
                accuracies.append(lab==labPred)
                self._logToCSV(
                    '', qn, 
                    self.classToAnsMap[labPred],
                    self.classToAnsMap[lab], 
                    labPred, lab, lab==labPred, img_id)
                
        valAcc = np.mean(accuracies)
        print('ValAcc: {:>6.1%}, total_preds: {}'.format(valAcc, total_predictions))
        return valAcc, correct_predictions, total_predictions
        
    
    def runTest(self, testReader, jsonOutputFile):
        '''For producing official test results for submission to server
        '''
        print('Starting test run...')
        allQnIds, allPreds = [], []
        for qnAsWordIDsBatch, seqLens, img_vecs, rawQns, img_ids, qn_ids \
            in testReader.getNextBatch(self.config.batch_size):
            feed = {
                self.word_ids : qnAsWordIDsBatch,
                self.sequence_lengths : seqLens,
                self.img_vecs : img_vecs,
                self.dropout : 1.0
            }
            labels_pred = self.sess.run(self.labels_pred, feed_dict=feed)
            
            for labPred, qn_id in zip(labels_pred, qn_ids):
                allQnIds.append(qn_id)
                allPreds.append(self.classToAnsMap[labPred])
        
        print('Total predictions: {}'.format(len(allPreds)))
        self._generateResultOutput(allQnIds, allPreds, jsonOutputFile)
        
    def _generateResultOutput(self, qn_ids, preds, jsonFile):
        '''
        result{
            "question_id": int,
            "answer": str
        }'''
        results = []
        for qn_id, pred in zip(qn_ids, preds):
            singleResult = {}
            singleResult["question_id"] = int(qn_id)
            singleResult["answer"] = str(pred)
            results.append(singleResult)
        
        with open(jsonFile, 'w') as jsonOut:
            print('Writing to {}'.format(jsonFile))
            json.dump(results, jsonOut)
        
    def destruct(self):
        pass
        
    