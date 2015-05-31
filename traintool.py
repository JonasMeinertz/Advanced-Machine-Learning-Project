import theano
import theano.tensor as T
import lasagne
import scipy.sparse
import numpy
import time
import datetime
import cPickle
from smtpmail import send_email
from loadtext import load_text

def train_model(model_out, data, name, learning_rate=0.01, batch_size=128):
    logname = datetime.datetime.now().strftime("%Y-%m-%d %H%M") + " - " + name
    n_epochs = 200

    train_set_x, train_set_y = data[0]
    valid_set_x, valid_set_y = data[1]
    test_set_x,  test_set_y  = data[2]

    n_train_batches = train_set_x.shape[0] / batch_size
    n_valid_batches = valid_set_x.shape[0] / batch_size
    n_test_batches  = test_set_x.shape[0] / batch_size

    print '... building the model'

    objective = lasagne.objectives.Objective(model_out,
        loss_function=lasagne.objectives.categorical_crossentropy)

    batch_index = T.iscalar('batch_index')
    X_batch = T.matrix('x')
    y_batch = T.ivector('y')
    batch_slice = slice(batch_index * batch_size, (batch_index + 1) * batch_size)

    loss_eval = objective.get_loss(X_batch, target=y_batch)

    pred = T.argmax(model_out.get_output(X_batch), axis=1)
    accuracy = T.mean(T.eq(pred, y_batch), dtype=theano.config.floatX)

    all_params = lasagne.layers.get_all_params(model_out)
    updates = lasagne.updates.momentum(loss_eval, all_params, learning_rate)

    train_model = theano.function(
        [X_batch, y_batch],
        loss_eval,
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
                send_email(logname, upd)

            if patience <= iter:
                done_looping = True
                break

    end_time = time.clock()
    upd = ('Optimization complete.\nBest validation score of %f %% obtained at iteration %i, '
           'with test performance %f %% \n The code for file '+os.path.split(__file__)[1]+' ran for %.2fm' %
           (best_validation_loss * 100., best_iter + 1, test_score * 100., (end_time - start_time) / 60.))
    print upd
    send_email(logname, upd)

    with open('parameters.pkl', 'w') as f:
        cPickle.dump(
            {'param_values': lasagne.layers.get_all_param_values(model_out)},
            f,
            -1 # for highest protocol
        )
