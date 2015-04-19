import theano
import theano.tensor as T
import lasagne
import numpy
import time
import os
import sys
from loadtext import load_text

def train_simple(learning_rate=0.01, n_epochs=200, nkerns=[50, 50], batch_size=128):
    rng = numpy.random.RandomState(5321)

    print '... loading text data'

    def shared_dataset(data_xy):
	    data_x, data_y = data_xy
	    shared_x = theano.shared(numpy.asarray(
	        data_x.reshape((data_x.shape[0], 1, 69, 130)),
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
    #print n_train_batches, n_valid_batches, n_test_batches

    print '... building the model'

    layer0_input = lasagne.layers.InputLayer(shape=(batch_size, 1, 69, 130))

    layer0 = lasagne.layers.Conv2DLayer(
        layer0_input,
        num_filters  = nkerns[0],
        filter_size  = (1, 5),
        nonlinearity = lasagne.nonlinearities.tanh
    )
    layer0_pool = lasagne.layers.MaxPool2DLayer(layer0, ds=(1, 3))

    layer1 = lasagne.layers.Conv2DLayer(
        layer0_pool,
        num_filters  = nkerns[0],
        filter_size  = (1, 3),
        nonlinearity = lasagne.nonlinearities.tanh
    )
    layer1_pool = lasagne.layers.MaxPool2DLayer(layer1, ds=(1, 4))

    layer2 = lasagne.layers.DenseLayer(
        layer1_pool,
        num_units = 500,
        nonlinearity = lasagne.nonlinearities.tanh
    )

    layer3 = lasagne.layers.DenseLayer(
        layer2,
        num_units = 14,
        nonlinearity = lasagne.nonlinearities.softmax
    )

    objective = lasagne.objectives.Objective(layer3,
        loss_function=lasagne.objectives.categorical_crossentropy)

    batch_index = T.iscalar('batch_index')
    X_batch = T.tensor4('x')
    y_batch = T.ivector('y')
    batch_slice = slice(batch_index * batch_size, (batch_index + 1) * batch_size)

    loss_eval = objective.get_loss(X_batch, target=y_batch)

    pred = T.argmax(layer3.get_output(X_batch), axis=1)
    accuracy = T.mean(T.eq(pred, y_batch), dtype=theano.config.floatX)

    all_params = lasagne.layers.get_all_params(layer3)
    updates = lasagne.updates.sgd(loss_eval, all_params, learning_rate)

    train_model = theano.function(
        [batch_index],
        loss_eval,
        updates = updates,
        givens = {
            X_batch: train_set_x[batch_slice],
            y_batch: train_set_y[batch_slice]
        }
    )

    validate_model = theano.function(
        [batch_index],
        [loss_eval, accuracy],
        givens = {
            X_batch: valid_set_x[batch_slice],
            y_batch: valid_set_y[batch_slice]
        }
    )

    test_model = theano.function(
        [batch_index],
        [loss_eval, accuracy],
        givens = {
            X_batch: test_set_x[batch_slice],
            y_batch: test_set_y[batch_slice]
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
                validation_losses = [validate_model(i)[1] for i
                                     in xrange(n_valid_batches)]

                this_validation_loss = 1-numpy.mean(validation_losses)
                print('epoch %i, minibatch %i/%i, validation error %f %%' %
                      (epoch, minibatch_index + 1, n_train_batches,
                       this_validation_loss * 100))

                if this_validation_loss < best_validation_loss:

                    if this_validation_loss < best_validation_loss * \
                       improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    test_losses = [test_model(i)[1] for i in xrange(n_test_batches)]
                    test_score = 1-numpy.mean(test_losses)
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
