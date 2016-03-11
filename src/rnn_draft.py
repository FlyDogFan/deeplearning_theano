import os
import sys
import timeit

import theano
import numpy
import theano.tensor as T

from logistic_sgd import LogisticRegression, load_data

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
        y_pred = T.argmax(self.output, axis=1)
        y_gold = T.argmax(y, axis=1)
        return T.mean(T.neq(y_pred, y_gold))
        
    def predict_y(self, raw_word):
        return numpy.argmax(raw_word, axis=1)
    
    def produce_y(self, index):
        y = []
        for i in range(self.n_words):
            if i == index:
                y.append(1)
            else:
                y.append(0)
        return y
        
    def predict(self, word, n_pred):
        
        k = T.iscalar("k")
        X = T.vector("X")

        def pre_step(s_t, x_t):
            s_t = T.tanh(T.dot(x_t, self.U) + T.dot(s_t, self.W))
            x_t = T.nnet.softmax(T.dot(s_t, self.V))
            return s_t, x_t[0]

        [s, x], updates = theano.scan(fn=pre_step,
                                      outputs_info=[self.s_init, X],
                                      n_steps=k)

        all_s = s
        all_x = x

        predict_model = theano.function(inputs=[X,k], outputs=all_x, updates=updates)
        
        y_pred = self.predict_y(numpy.asarray(predict_model(word, n_pred)))
        
        y_list = []
        
        for each_y in y_pred:
            y_list.append(self.produce_y(each_y))
        
        return y_list
        
def test_RNN(learning_rate = 0.1, batch_size = 2):
    
    sentences = theano.shared(
        numpy.asarray(
            [[[0,0,1,0],[0,1,0,0],[1,0,0,0]],
             [[0,1,0,0],[1,0,0,0],[0,1,0,0]]],
            dtype = theano.config.floatX
        ),
        borrow = True
    )
    labels = theano.shared(
        numpy.asarray(
            [[[0,1,0,0],[1,0,0,0],[0,0,0,1]],
             [[1,0,0,0],[0,1,0,0],[0,0,1,0]]],
            dtype = theano.config.floatX
        ),
        borrow = True
    )
    
    n_train = sentences.get_value().shape[0]
    
    print "... building model"
    
    index = T.lscalar('index')
    x = T.matrix('x') 
    y = T.matrix('y')
    
    rng = numpy.random.RandomState(23455)
    
    my_rnn = RNNLayer(
        rng = rng,
        input = x,
        n_words = 4,
        n_hidden = 10,
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
            x: sentences[index],
            y: labels[index]
        }
        #on_unused_input='ignore'
    )
    
    print "... training model"
    
    for i in range(2000):
        for j in range(n_train):
            loss = train_model(j)
    print loss
    
    word = numpy.asarray([0,0,1,0])
    
    pred_y_list = my_rnn.predict(word = word, n_pred = 3)
    
    print pred_y_list
    
    """
    sentence = T.matrix('sentence')
    
    words_pred = my_rnn.predict(sentence = sentence, n_pred = 3)
    
    pred_model = theano.function(
        inputs = [sentence],
        outputs = words_pred
    )
    
    
    
    pred_model(sent)
    """
        
if __name__ == '__main__':
    test_RNN()