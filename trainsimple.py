import theano
import theano.tensor as T
import numpy
import time
import os

import sys
from loadtext import load_text
sys.path.append('tutorials')
from logistic_sgd import LogisticRegression
from mlp import HiddenLayer
from convolutional_mlp import LeNetConvPoolLayer

def train_simple(learning_rate=0.03, n_epochs=200, nkerns=[50, 50], batch_size=128):
    rng = numpy.random.RandomState(5321)

    print '... loading text data'

    def shared_dataset(data_xy):
	    data_x, data_y = data_xy
	    shared_x = theano.shared(numpy.asarray(
	        data_x,
	        dtype = theano.config.floatX
	    ), borrow=True)
	    shared_y = theano.shared(numpy.asarray(
	        data_y,
	        dtype = theano.config.floatX
	    ))
	    return shared_x, T.cast(shared_y, 'int32')

    datasets = load_text(130, rng)

    train_set_x, train_set_y = shared_dataset(datasets[0])
    valid_set_x, valid_set_y = shared_dataset(datasets[1])
    test_set_x, test_set_y = shared_dataset(datasets[2])

    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / batch_size
    n_test_batches  = test_set_x.get_value(borrow=True).shape[0] / batch_size

    print '... building the model'

    index = T.lscalar()
    x = T.matrix('x')
    y = T.ivector('y')

    layer0_input = x.reshape((batch_size, 1, 69, 130))

    layer0 = LeNetConvPoolLayer(
        rng,
        input        = layer0_input,
        image_shape  = (batch_size, 1, 69, 130),
        filter_shape = (nkerns[0], 1, 1, 5),
        poolsize     = (1, 3)
    )

    layer1 = LeNetConvPoolLayer(
        rng,
        input        = layer0.output,
        image_shape  = (batch_size, nkerns[0], 69, 42),
        filter_shape = (nkerns[1], nkerns[0], 1, 3),
        poolsize     = (1, 4)
    )

    layer2_input = layer1.output.flatten(2)

    layer2 = HiddenLayer(
        rng,
        input      = layer2_input,
        n_in       = nkerns[1] * 69 * 10,
        n_out      = 500,
        activation = T.tanh
    )

    layer3 = LogisticRegression(input=layer2.output, n_in=500, n_out=14)

    test_model = theano.function(
        [index],
        layer3.errors(y),
        givens = {
            x: test_set_x[index * batch_size : (index + 1) * batch_size],
            y: test_set_y[index * batch_size : (index + 1) * batch_size]
        }
    )

    validate_model = theano.function(
        [index],
        layer3.errors(y),
        givens = {
            x: valid_set_x[index * batch_size : (index + 1) * batch_size],
            y: valid_set_y[index * batch_size : (index + 1) * batch_size]
        }
    )

    cost = layer3.negative_log_likelihood(y)
    params = layer3.params + layer2.params + layer1.params + layer0.params
    grads = T.grad(cost, params)
    updates = [
        (param_i, param_i - learning_rate * grad_i)
        for param_i, grad_i in zip(params, grads)
    ]

    train_model = theano.function(
        [index],
        cost,
        updates = updates,
        givens = {
            x: train_set_x[index * batch_size : (index + 1) * batch_size],
            y: train_set_y[index * batch_size : (index + 1) * batch_size]
        }
    )

    print '... training the model'

    patience = 10000
    patience_increase = 2
    improvement_threshold = 0.995
    validation_frequency = min(n_train_batches, patience / 2)

    best_validation_loss = numpy.inf
    best_iter = 0
    test_score = 0.
    start_time = time.clock()

    epoch = 0
    done_looping = False

    while (epoch < n_epochs) and (not done_looping):
        epoch += 1
        for minibatch_index in xrange(n_train_batches):
            iter = (epoch - 1) * n_train_batches + minibatch_index

            if iter % 100 == 0:
                print 'training @ iter = ', iter
            train_model(minibatch_index)

            if (iter + 1) % validation_frequency == 0:
                validation_losses = [validate_model(i) for i
                                     in xrange(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)
                print('epoch %i, minibatch %i/%i, validation error %f %%' %
                      (epoch, minibatch_index + 1, n_train_batches,
                       this_validation_loss * 100))

                if this_validation_loss < best_validation_loss:

                    if this_validation_loss < best_validation_loss * \
                       improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    test_losses = [test_model(i) for i in xrange(n_test_batches)]
                    test_score = numpy.mean(test_losses)
                    print(('    epoch %i, minibatch %i/%i, test error of '
                           'best model %f %%') %
                          (epoch, minibatch_index + 1, n_train_batches,
                           test_score * 100))

            if patience <= iter:
                done_looping = True
                break

    end_time = time.clock()
    print 'Optimization complete.'
    print ('Best validation score of %f %% obtained at iteration %i, '
           'with test performance %f %%' %
           (best_validation_loss * 100., best_iter + 1, test_score * 100.))
    print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))

if __name__ == "__main__":
    train_simple()
