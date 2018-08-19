# -*- coding: utf-8 -*-
"""
Created on Wed Jul  4 17:47:55 2018

@author: hsasa
"""

from nltk.tokenize import sent_tokenize
import keras
import numpy as np
from keras.models import Model, load_model
from keras.layers import Dense, Input, Dropout, MaxPooling1D, Conv1D, GlobalMaxPool1D, Bidirectional
from keras.layers import LSTM, Lambda, Bidirectional, concatenate, BatchNormalization, Embedding
from keras.layers import TimeDistributed
from keras.optimizers import Adam
import tensorflow as tf
import keras.backend as K

class CharCNN:
    
    
    CHAR_DICT = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .!?:,\'%-\(\)/$|&;[]"'
    
    def __init__(self, max_len_s, max_num_s, verbose=10):
        self.max_len_s = max_len_s
        self.max_num_s = max_num_s
        self.verbose =verbose
        self.num_of_char = 0
        self.num_of_label = 0
        self.unknown_label = ''
        
    def build_dic(self, char_dict=None, unknown_label = 'UNK'):
        if char_dict is None:
            char_dict = self.CHAR_DICT
            
        self.unknown_label = unknown_label
        
        char = []
        
        for c in char_dict:
            char.append(c)
        
        char = list(set(char))
        
        char.insert(0, unknown_label)
        
        self.num_of_char = len(char)
        self.char_indices = dict((c,i) for i, c in enumerate(char))
        self.indices_char = dict((i,c) for i, c in enumerate(char))
        
        return self.char_indices, self.indices_char, self.num_of_char
    
    def convert_labels(self, labels):
        
        self.label2indexes = dict((l,i) for i, l in enumerate(labels))
        self.index2labels = dict((i,l) for i, l in enumerate(labels))
        self.num_of_label = len(self.label2indexes)
        return self.label2indexes, self.index2labels
    
    def transormdata(self, df, x_col, y_col, label2indexes=None, sample_size=None):
        x=[]
        y=[]
        actual_max_sentence = 0
        if sample_size is None:
            sample_size = len(df)
        
        for i, row in df.head(sample_size).iterrows():
            x_data = row[x_col]
            y_data = row[y_col]
            
            
            sentences = sent_tokenize(x_data)
            x.append(sentences)
            
            if len(sentences) > actual_max_sentence:
                actual_max_sentence = len(sentences)
            
            y.append(label2indexes[y_data])
            
        
        return x, y
    
    def transform_data_train(self, x_raw, y_raw, max_len_s = None, max_num_s=None):
        
        unknow = self.char_indices[self.unknown_label]
        
        x = np.ones((len(x_raw), max_num_s, max_len_s),dtype=np.int64)*unknow
        y = np.array(y_raw)
        
        if max_len_s is None:
            max_len_s = self.max_len_s
        if max_num_s is None:
            max_num_s = self.max_num_s
        
        for i, doc in enumerate(x_raw):
            for j, sentence in enumerate(doc):
                if j < max_num_s:
                    for t, char in enumerate(sentence[-max_len_s:]):
                        if char not in self.char_indices:
                            x[i, j, (max_len_s-1-t)] = self.char_indices['UNK']
                        else:
                            x[i, j, (max_len_s-1-t)] = self.char_indices[char]

        return x, y
    
    def build_char_block(self, block, dropout = 0.3, filters=[64,100], kernel_size = [3,3], pool_size = [2,2], padding = 'valid', activation = 'relu', kernel_initializer='glorot_normal' ):
        for i in range(len(filters)):
            block = Conv1D(filters = filters[i], kernel_size= kernel_size[i], padding = padding, activation = activation, kernel_initializer = kernel_initializer)(block)
            
        block = Dropout(dropout)(block)
        block = MaxPooling1D(pool_size = pool_size[i])(block)
        
        block = GlobalMaxPool1D()(block)
        block = Dense(128, activation = 'relu')(block)
        return block
    
    def build_sentence_block(self, max_len_s, max_num_s, 
                              char_dimension=16,
                              filters=[[3, 5, 7], [200, 300, 300], [300, 400, 400]],  
                              kernel_sizes=[[4, 3, 3], [5, 3, 3], [6, 3, 3]], 
                              pool_sizes=[[2, 2, 2], [2, 2, 2], [2, 2, 2]],
                              dropout=0.4):
        sent_input = Input(shape=(max_len_s, ),dtype='int64')
        embedded = Embedding(self.num_of_char, char_dimension, input_length = max_len_s)(sent_input)
        
        blocks=[]
        for i, filter_layers in enumerate(filters):
            blocks.append(
                self.build_char_block(
                    block=embedded, filters=filters[i], kernel_size=kernel_sizes[i], pool_size=pool_sizes[i])
            )
        sent_output = concatenate(blocks, axis=-1)
        sent_output = Dropout(dropout)(sent_output)
        sent_encoder = Model(inputs=sent_input, outputs=sent_output)

        return sent_encoder
    
    def build_doc_block(self, sent_encoder, max_len_s, max_num_s, 
                             num_of_label, dropout=0.3, 
                             loss='sparse_categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy']):
        
        doc_input = Input(shape=(max_num_s, max_len_s), dtype='int64')
        doc_output = TimeDistributed(sent_encoder)(doc_input)

        doc_output = Bidirectional(LSTM(128, return_sequences=False, dropout=dropout))(doc_output)

        doc_output = Dropout(dropout)(doc_output)
        doc_output = Dense(128, activation='relu')(doc_output)
        doc_output = Dropout(dropout)(doc_output)
        doc_output = Dense(num_of_label, activation='sigmoid')(doc_output)

        doc_encoder = Model(inputs=doc_input, outputs=doc_output)
        doc_encoder.compile(loss=loss, optimizer=optimizer, metrics=metrics)
        return doc_encoder
   
    def preporcess(self, labels, char_dict=None, unknown_label='UNK'):
        self.build_dic(char_dict, unknown_label)
        self.convert_labels(labels)
        
    def process(self, df, x_col, y_col, max_len_s=None, max_num_s=None, label2indexes=None, sample_size=None):
        if sample_size is None:
            sample_size=1000
        if label2indexes is None:
            if self.label2indexes is None:
                raise Exception('Run PreProcess First')
            label2indexes = self.label2indexes
        if max_len_s is None:
            max_len_s = self.max_len_s
        if max_num_s is None:
            max_num_s = self.max_num_s
            
            
        x_preprocess, y_preprocess = self.transormdata(df = df, x_col = x_col, y_col = y_col, label2indexes=label2indexes)
        x_preprocess, y_preprocess = self.transform_data_train(x_raw=x_preprocess, y_raw=y_preprocess,max_len_s=max_len_s,max_num_s=max_num_s)
        
        return x_preprocess, y_preprocess
    
    def build_model(self, char_dimension = 16, display_summary = False, dispaly_architecture = False, loss='sparse_categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy']):
        sent_encoder = self.build_sentence_block(char_dimension=char_dimension,max_len_s=self.max_len_s,max_num_s=self.max_num_s)
        doc_encoder = self.build_doc_block(
            sent_encoder=sent_encoder, num_of_label=self.num_of_label,
            max_len_s=self.max_len_s, max_num_s=self.max_num_s, 
            loss=loss, optimizer=optimizer, metrics=metrics)
        self.model = {'sent_encoder': sent_encoder,
            'doc_encoder': doc_encoder
        }
        
        return doc_encoder
    
    def train(self, x_train, y_train, x_test, y_test, batch_size=128, epochs=1, shuffle=True):
        self.get_model().fit(
            x_train, y_train, validation_data=(x_test, y_test), 
            batch_size=batch_size, epochs=epochs, shuffle=shuffle)
    
    def predict(self, x, model=None, return_prob=False):
        if model is None:
            model = self.get_model()
            
        if return_prob:
            return model.predict(x)
        
        return model.predict(x).argmax(axis=-1)
        
    def get_model(self):
        return self.model['doc_encoder']
    
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        