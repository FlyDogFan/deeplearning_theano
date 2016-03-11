# -*- coding: utf-8 -*-

__author__ = "tianwen jiang"

import os
import sys
import timeit

import theano
import numpy
import theano.tensor as T

"""
from pyltp import Segmentor, Postagger, NamedEntityRecognizer

MODELDIR="/data/ltp/ltp-models/3.3.0/ltp_data"
segmentor = Segmentor()
segmentor.load(os.path.join(MODELDIR, "cws.model"))
"""

class RNNLayer(object):
    def __init__(self, rng, input, n_words, n_hidden):
        
        self.input = input
        
        U_bound = numpy.sqrt(6.0 / (n_words + n_hidden))
        self.U = theano.shared(
            numpy.asarray(
                rng.uniform(
                    low = -U_bound,
                    high = U_bound,
                    size = (n_words, n_hidden)
                ),
                dtype = theano.config.floatX
            ),
            borrow = True
        )
        
        W_bound = numpy.sqrt(6.0 / (n_hidden + n_hidden))
        self.W = theano.shared(
            numpy.asarray(
                rng.uniform(
                    low = -W_bound,
                    high = W_bound,
                    size = (n_hidden, n_hidden)
                ),
                dtype = theano.config.floatX
            ),
            borrow = True
        )
        
        V_bound = numpy.sqrt(6.0 / (n_hidden + n_words))
        self.V = theano.shared(
            numpy.asarray(
                rng.uniform(
                    low = -V_bound,
                    high = V_bound,
                    size = (n_hidden, n_words)
                ),
                dtype = theano.config.floatX
            ),
            borrow = True
        )
        
        self.n_words = n_words
        self.n_hidden = n_hidden
        
        self.params = [self.U, self.W, self.V]
        
        def step(x_t, s_pre):
            s_t = T.tanh(T.dot(x_t, self.U) + T.dot(s_pre, self.W))
            o_t = T.nnet.softmax(T.dot(s_t, self.V))
            return s_t, o_t[0]
        
        init = numpy.zeros((n_hidden,), dtype=theano.config.floatX)
        self.s_init = theano.shared(value=init, name='s_init')
        
        [s_list, o_list], updates = theano.scan(step,
            sequences=self.input,
            outputs_info=[self.s_init, None]
        )
        
        self.output = o_list
    
    def calculate_loss(self, y):
        return T.mean(T.nnet.categorical_crossentropy(self.output, y)) 
    
    def error(self, y):
        words_pred = T.argmax(self.output, axis=1)
        words_gold = T.argmax(y, axis=1)
        return T.mean(T.neq(words_pred, words_gold))
        
    def predict_words(self, y_pred):
        return numpy.argmax(y_pred, axis=1)
    
    def produce_y(self, word):
        y = []
        for i in range(self.n_words):
            if i == word:
                y.append(1)
            else:
                y.append(0)
        return y
        
    def predict(self, word_vector, n_pred):
        
        k = T.iscalar("k")
        w = T.vector("w")

        def pre_step(s_t, x_t):
            s_t = T.tanh(T.dot(x_t, self.U) + T.dot(s_t, self.W))
            x_t = T.nnet.softmax(T.dot(s_t, self.V))
            return s_t, x_t[0]

        [s, x], updates = theano.scan(fn=pre_step,
                                      outputs_info=[self.s_init, w],
                                      n_steps=k)

        predict_model = theano.function(inputs=[w,k], outputs=x, updates=updates)
        
        words_pred = self.predict_words(numpy.asarray(predict_model(word_vector, n_pred)))
        
        return words_pred
        
def test_RNN(learning_rate=0.13, n_epochs=1000,
                           dataset='wangfeng.txt'):
    
    print "... loading data"
    
    train_set_x, train_set_y, vocabulary = load_data(dataset)
    
    n_train = train_set_x.get_value().shape[0]
    
    print "... building model"
    
    index = T.lscalar('index')
    x = T.matrix('x') 
    y = T.matrix('y')
    
    rng = numpy.random.RandomState(23455)
    
    my_rnn = RNNLayer(
        rng = rng,
        input = x,
        n_words = len(vocabulary),
        n_hidden = 100,
    )
    
    params = my_rnn.params
    error = my_rnn.error(y)
    cost = my_rnn.calculate_loss(y)
    grads = T.grad(cost, params)  
    updates = [(param_i, param_i-learning_rate*grad_i) for param_i, grad_i in zip (params, grads)]
    
    train_model = theano.function(
        inputs = [index], 
        outputs = cost, 
        updates=updates,
        givens = {
            x: train_set_x[index],
            y: train_set_y[index]
        }
        #on_unused_input='ignore'
    )
    
    print "... training model"
    
    for echo in range(n_epochs):
        for j in range(n_train):
            loss = train_model(j)
            print 'echo:%d\tloss:%f' % (echo, loss)
    
    w_index = vocabulary.index(u'我')
    w_vector = [0.0 for j in range(len(vocabulary))]
    w_vector[w_index] = 1.0
    word_vector = numpy.asarray(w_vector, dtype=theano.config.floatX)
    
    words_pred = my_rnn.predict(word_vector = word_vector, n_pred = 6)
    
    print u'我',
    
    for word_pred in words_pred:
        print vocabulary[word_pred],
    
def load_data(dataset):
    
    data_dir, data_file = os.path.split(dataset)
    if data_dir == "" and not os.path.isfile(dataset):
        new_path = os.path.join(
            os.path.split(__file__)[0],
            "..",
            "data",
            dataset
        )
        if os.path.isfile(new_path) or data_file == 'mnist.pkl.gz':
            dataset = new_path
    
    file = open(dataset, 'r')
    
    sentences = []
    vocabulary = set()
    line = file.readline()
    while line:
        sentence = line.strip().decode('utf-8')
        if sentence == '':
            line = file.readline()
            continue
        #words = segmentor.segment(sentence)
        #print ' '.join(words)
        words = sentence
        sentences.append(sentence)
        for word in sentence:
            vocabulary.add(word)
        line = file.readline()
    
    vocabulary = list(vocabulary)
    train_set_x_value = []
    train_set_y_value = []
    
    for sentence in sentences:
        s_v = []
        for i in range(len(sentence)):
            #try:
            index = vocabulary.index(sentence[i])
            vector = [0 for j in range(len(vocabulary))]
            vector[index] = 1
            s_v.append(vector)
            # except:
                # print "word2vec error."
                # sys.exit(1)
        train_set_x_value.append(s_v[:-1])
        train_set_y_value.append(s_v[1:])
    
    train_set_x = theano.shared(
        numpy.asarray(
            train_set_x_value[0:2],
            dtype = theano.config.floatX
        ),
        borrow = True
    )
    train_set_y = theano.shared(
        numpy.asarray(
            train_set_y_value[0:2],
            dtype = theano.config.floatX
        ),
        borrow = True
    )
    
    return (train_set_x, train_set_y, vocabulary)
    
if __name__ == '__main__':
    test_RNN()