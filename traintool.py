import theano
import theano.tensor as T
import lasagne
import scipy.sparse
import numpy
import os
import time
import datetime
import cPickle
from smtpmail import send_email
from loadtext import load_text

def train_model(model_out, data, name, learning_rate_schedule={0: 0.01},
                batch_size=128, n_epochs=200):
    logname = datetime.datetime.now().strftime("%Y-%m-%d %H%M") + " - " + name

    # prepare data
    train_set_x, train_set_y = data[0]
    valid_set_x, valid_set_y = data[1]
    test_set_x,  test_set_y  = data[2]
    n_train_batches = train_set_x.shape[0] / batch_size
    n_valid_batches = valid_set_x.shape[0] / batch_size
    n_test_batches  = test_set_x.shape[0] / batch_size

    print '... building the model'
    print "Total parameters: {}".format(
    sum([p.get_value().size for p in lasagne.layers.get_all_params(model_out)]))

    # define symbolic inputs
    batch_index = T.iscalar('batch_index')
    X_batch = T.matrix('x')
    y_batch = T.ivector('y')
    batch_slice = slice(batch_index * batch_size, (batch_index + 1) * batch_size)

    # learning rate as sared variable
    learning_rate = theano.shared(numpy.array(learning_rate_schedule[0],
                                  dtype=theano.config.floatX)
                                 )

    # evaluate objective loss function
    objective = lasagne.objectives.Objective(model_out,
        loss_function=lasagne.objectives.categorical_crossentropy)
    loss_eval = objective.get_loss(X_batch, target=y_batch)

    # evaluate predictions and accuracy
    pred = T.argmax(lasagne.layers.get_output(model_out, X_batch, deterministic=True), axis=1)
    accuracy = T.mean(T.eq(pred, y_batch), dtype=theano.config.floatX)

    # compute gradients
    all_params = lasagne.layers.get_all_params(model_out)
    updates = lasagne.updates.momentum(loss_eval, all_params, learning_rate)

    # compile functions
    train_model = theano.function(
        [X_batch, y_batch],
        loss_eval,
        updates = updates,
        allow_input_downcast=True
    )

    eval_model = theano.function(
        [X_batch, y_batch],
        [accuracy, pred],
        allow_input_downcast=True
    )

    # helper function to get a chunk of data
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
    losses = []

    epoch = 0
    done_looping = False

    while (epoch < n_epochs) and (not done_looping):
        if epoch in learning_rate_schedule:
            current_lr = learning_rate_schedule[epoch]
            learning_rate.set_value(learning_rate_schedule[epoch])
            print "  setting learning rate to %.6f" % current_lr

        epoch += 1
        for minibatch_index in xrange(n_train_batches):
            iter = (epoch - 1) * n_train_batches + minibatch_index

            if iter % 100 == 0:
                print 'training @ iter = ', iter
            (X_bat, y_bat) = get_batch(minibatch_index, train_set_x, train_set_y)
            loss = train_model(X_bat, y_bat)
            losses.append(loss)

            if (iter + 1) % validation_frequency == 0:
                validation_losses = range(n_valid_batches)
                for i in validation_losses:
                    (X_bat, y_bat) = get_batch(i, valid_set_x, valid_set_y)
                    validation_losses[i], predictions = eval_model(X_bat, y_bat)

                this_validation_loss = 1-numpy.mean(validation_losses)
                mean_train_loss = numpy.mean(losses)

                if this_validation_loss < best_validation_loss:

                    if this_validation_loss < best_validation_loss * \
                       improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    best_validation_loss = this_validation_loss
                    best_iter = iter
                    confusion_matrix = numpy.zeros([14,14], dtype=numpy.int_)

                    test_losses = range(n_test_batches)
                    for i in test_losses:
                        (X_bat, y_bat) = get_batch(i, test_set_x, test_set_y)
                        test_losses[i], predictions = eval_model(X_bat, y_bat)
                        for prediction, target in zip(predictions, y_bat):
                            confusion_matrix[target][prediction] += 1
                    test_score = 1-numpy.mean(test_losses)
                    send_email(logname+' confusion', numpy.array_str(confusion_matrix))

                upd = ', '.join([
                    str(time.clock() - start_time),
                    str(iter),
                    str(patience),
                    str(epoch),
                    str(minibatch_index),
                    str(mean_train_loss),
                    str(this_validation_loss),
                    str(test_score)
                ])
                print upd
                send_email(logname, upd)

            if patience < iter:
                done_looping = True
                break

    upd = 'training finished!'
    print upd
    send_email(logname, upd)

    parameter_file = 'parameters/' + name + '.pkl'
    with open(parameter_file, 'w') as f:
        cPickle.dump(
            {'param_values': lasagne.layers.get_all_param_values(model_out)},
            f,
            -1 # for highest protocol
        )
