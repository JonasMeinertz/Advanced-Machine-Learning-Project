import theano
import theano.tensor as T
import lasagne
import scipy.sparse
import numpy
import time
import os
import sys
import cPickle
from smtpmail import send_email
from loadtextsparse import load_text

def train_simple(n_epochs=200, nkerns=200, batch_size=128):
    rng = numpy.random.RandomState(5321)

    LEARNING_RATE_SCHEDULE = {
        0: 0.01,
        3: 0.005,
        6: 0.0025,
        9: 0.00125,
        12: 0.000625,
        15: 0.0003125,
        18: 0.00015625,
        21: 0.000078125,
        24: 0.0000390625,
        27: 0.00001953125,
        30: 0.000009765625
    }

    print '... loading text data'

    def shared_dataset(data_y):
	    #data_x, data_y = data_xy
	    #shared_x = theano.shared(data_x)
	    shared_y = theano.shared(numpy.asarray(
	        data_y,
	        dtype = theano.config.floatX
	    ))
	    return T.cast(shared_y, 'int32')

    train_set = load_text(data='data/dbpedia_csv/train.csv', samples=560000, length=1014)
    test_set = load_text(data='data/dbpedia_csv/test.csv', samples=70000, length=1014)

    train_set_x = train_set[0]
    train_set_y = train_set[1]
    valid_set_x = test_set[0][0:35000]
    valid_set_y = test_set[1][0:35000]
    test_set_x  = test_set[0][35000:70000]
    test_set_y  = test_set[1][35000:70000]

    n_train_batches = train_set_x.shape[0] / batch_size
    n_valid_batches = valid_set_x.shape[0] / batch_size
    n_test_batches  = test_set_x.shape[0] / batch_size

    print '... building the model'

    layer0_flat = lasagne.layers.InputLayer(shape=(batch_size, 69 * 1014))
    layer0_input = lasagne.layers.ReshapeLayer(layer0_flat, (batch_size, 1, 69, 1014))

    layer1 = lasagne.layers.Conv2DLayer(
        layer0_input,
        num_filters  = nkerns,
        filter_size  = (1, 7),
        nonlinearity = lasagne.nonlinearities.rectify,
        W            = lasagne.init.Normal(0.05, 0)
    )
    layer1_pool = lasagne.layers.MaxPool2DLayer(layer1, pool_size=(1, 3))

    layer2 = lasagne.layers.Conv2DLayer(
        layer1_pool,
        num_filters  = nkerns,
        filter_size  = (1, 7),
        nonlinearity = lasagne.nonlinearities.rectify,
        W            = lasagne.init.Normal(0.05, 0)
    )
    layer2_pool = lasagne.layers.MaxPool2DLayer(layer2, pool_size=(1, 3))

    layer3 = lasagne.layers.Conv2DLayer(
        layer2_pool,
        num_filters  = nkerns,
        filter_size  = (1, 3),
        nonlinearity = lasagne.nonlinearities.rectify,
        W            = lasagne.init.Normal(0.05, 0)
    )

    layer4 = lasagne.layers.Conv2DLayer(
        layer3,
        num_filters  = nkerns,
        filter_size  = (1, 3),
        nonlinearity = lasagne.nonlinearities.rectify,
        W            = lasagne.init.Normal(0.05, 0)
    )

    layer5 = lasagne.layers.Conv2DLayer(
        layer4,
        num_filters  = nkerns,
        filter_size  = (1, 3),
        nonlinearity = lasagne.nonlinearities.rectify,
        W            = lasagne.init.Normal(0.05, 0)
    )

    layer6 = lasagne.layers.Conv2DLayer(
        layer5,
        num_filters  = nkerns,
        filter_size  = (1, 3),
        nonlinearity = lasagne.nonlinearities.rectify,
        W            = lasagne.init.Normal(0.05, 0)
    )
    layer6_pool = lasagne.layers.MaxPool2DLayer(layer6, pool_size=(1, 3))

    layer7 = lasagne.layers.DenseLayer(
        layer6_pool,
        num_units    = 1024,
        nonlinearity = lasagne.nonlinearities.rectify,
        W            = lasagne.init.Normal(0.05, 0)
    )

    dropout1 = lasagne.layers.DropoutLayer(layer7)

    layer8 = lasagne.layers.DenseLayer(
        dropout1,
        num_units    = 1024,
        nonlinearity = lasagne.nonlinearities.rectify,
        W            = lasagne.init.Normal(0.05, 0)
    )

    dropout2 = lasagne.layers.DropoutLayer(layer8)

    layer9 = lasagne.layers.DenseLayer(
        dropout2,
        num_units    = 14,
        nonlinearity = lasagne.nonlinearities.softmax,
        W            = lasagne.init.Normal(0.05, 0)
    )

    objective = lasagne.objectives.Objective(layer9,
        loss_function=lasagne.objectives.categorical_crossentropy)

    batch_index = T.iscalar('batch_index')
    X_batch = T.matrix('x')
    y_batch = T.ivector('y')

    learning_rate = theano.shared(numpy.array(LEARNING_RATE_SCHEDULE[0], dtype=theano.config.floatX))

    loss_train = objective.get_loss(X_batch, target=y_batch)
    all_params = lasagne.layers.get_all_params(layer9)
    updates = lasagne.updates.momentum(loss_train, all_params, learning_rate, momentum=0.9)

    loss_eval = objective.get_loss(X_batch, target=y_batch, deterministic=True)
    pred = T.argmax(lasagne.layers.get_output(layer9, X_batch, deterministic=True), axis=1)
    accuracy = T.mean(T.eq(pred, y_batch), dtype=theano.config.floatX)

    train_model = theano.function(
        [X_batch, y_batch],
        loss_train,
        updates = updates,
        allow_input_downcast=True
    )

    eval_model = theano.function(
        [X_batch, y_batch],
        accuracy,
        allow_input_downcast=True
    )

    def get_batch(index, set_x, set_y):
        X_bat = (set_x[index * batch_size : (index + 1) * batch_size]).todense()
        y_bat =  set_y[index * batch_size : (index + 1) * batch_size]
        return X_bat, y_bat

    print '... training the model'

    patience = 10000
    patience_increase = 1.5
    improvement_threshold = 0.995
    validation_frequency = min(n_train_batches, patience / 2)

    best_validation_loss = numpy.inf
    best_iter = 0
    test_score = 0.
    start_time = time.clock()

    epoch = 0
    done_looping = False

    while (epoch < n_epochs) and (not done_looping):
        if epoch in LEARNING_RATE_SCHEDULE:
            current_lr = LEARNING_RATE_SCHEDULE[epoch]
            learning_rate.set_value(LEARNING_RATE_SCHEDULE[epoch])
            print "  setting learning rate to %.6f" % current_lr

        epoch += 1
        for minibatch_index in xrange(n_train_batches):
            iter = (epoch - 1) * n_train_batches + minibatch_index

            if iter % 100 == 0:
                print 'training @ iter = ', iter
            (X_bat, y_bat) = get_batch(minibatch_index, train_set_x, train_set_y)
            train_model(X_bat, y_bat)

            if (iter + 1) % validation_frequency == 0:
                validation_losses = range(n_valid_batches)
                for i in validation_losses:
                    (X_bat, y_bat) = get_batch(i, valid_set_x, valid_set_y)
                    validation_losses[i] = eval_model(X_bat, y_bat)

                this_validation_loss = 1-numpy.mean(validation_losses)

                if this_validation_loss < best_validation_loss:

                    if this_validation_loss < best_validation_loss * \
                       improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    test_losses = range(n_test_batches)
                    for i in test_losses:
                        (X_bat, y_bat) = get_batch(i, test_set_x, test_set_y)
                        test_losses[i] = eval_model(X_bat, y_bat)
                    test_score = 1-numpy.mean(test_losses)

                upd = ' '.join([str(time.clock() - start_time), str(iter), str(patience), str(epoch), str(minibatch_index), str(this_validation_loss), str(test_score)])
                print upd
                send_email('fullmodel', upd)

            if patience <= iter:
                done_looping = True
                break

    end_time = time.clock()
    upd = ('Optimization complete.\nBest validation score of %f %% obtained at iteration %i, '
           'with test performance %f %% \n The code for file '+os.path.split(__file__)[1]+' ran for %.2fm' %
           (best_validation_loss * 100., best_iter + 1, test_score * 100., (end_time - start_time) / 60.))
    print upd
    send_email('fullmodel', upd)

    with open('parameters.pkl', 'w') as f:
        cPickle.dump(
            {'param_values': lasagne.layers.get_all_param_values(layer3)},
            f,
            -1 # for highest protocol
        )

    os.system('shutdown now -h')

if __name__ == "__main__":
    train_simple()
