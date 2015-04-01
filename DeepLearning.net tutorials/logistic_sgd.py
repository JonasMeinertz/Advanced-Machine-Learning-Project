import cPickle
import gzip
import os
import sys
import time

import theano
import theano.tensor as T

import numpy
# code taken from
# http://deeplearning.net/tutorial/logreg.html#creating-a-logisticregression-class
# look there for a better commented version of the code

class LogisticRegression(object):
	def __init__(self, input, n_in, n_out):

		# define model parameters as shared values and initialize with zeros
		self.W = theano.shared(
			value  = numpy.zeros((n_in,n_out), dtype=theano.config.floatX),
			name   = 'W',
			borrow = True
		)
		self.b = theano.shared(
			value  = numpy.zeros(n_out, dtype=theano.config.floatX),
			name   = 'b',
			borrow = True
		)

		# symbolic expression for computing the matrix of class membership
		# probabilities
		self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)
		self.y_pred = T.argmax(self.p_y_given_x, axis=1)
		self.params = [self.W, self.b]

	def negative_log_likelihood(self, y):
		return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

	def errors(self, y):
		if y.ndim != self.y_pred.ndim:
			raise TypeError(
				'y should have the same shape as self.y_pred',
				('y', y.type, 'y_pred', self.y_pred.type)
			)
		if y.dtype.startswith('int'):
			return T.mean(T.neq(self.y_pred, y))
		else:
			raise NotImplementedError()

def load_data(dataset):
	print '... loading data'
	f = gzip.open('../mnist.pkl.gz', 'rb')
	train_set, valid_set, test_set = cPickle.load(f)
	f.close()

	def shared_dataset(data_xy):
	    data_x, data_y = data_xy
	    shared_x = theano.shared(numpy.asarray(
	        data_x,
	        dtype = theano.config.floatX
	    ))
	    shared_y = theano.shared(numpy.asarray(
	        data_y,
	        dtype = theano.config.floatX
	    ))
	    return shared_x, T.cast(shared_y, 'int32')

	train_set_x, train_set_y = shared_dataset(train_set)
	valid_set_x, valid_set_y = shared_dataset(valid_set)
	test_set_x,  test_set_y  = shared_dataset(test_set)

	rval = [(train_set_x, train_set_y),
			(valid_set_x, valid_set_y),
			(test_set_x, test_set_y)]
	return rval

def sgd_optimization_mnist(learning_rate = 0.13,
						   n_epochs = 1000,
						   dataset='../mnist.pkl.gz',
						   batch_size=600):
	datasets = load_data(dataset)
	train_set_x, train_set_y = datasets[0]
	valid_set_x, valid_set_y = datasets[1]
	test_set_x,  test_set_y  = datasets[2]

	# compute number of minibatches for training, validation and testing
	n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
	n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / batch_size
	n_test_batches = test_set_x.get_value(borrow=True).shape[0] / batch_size

	######################
	# BUILD ACTUAL MODEL #
	######################
	print '... building the model'

	# allocate symbolic variables for the data
	index = T.lscalar() # index to a [mini]batch

	# generate symbolic variables for input (x and y represent a minibatch)
	x = T.matrix('x')
	y = T.ivector('y')

	# construct logistic regression classifier
	classifier = LogisticRegression(input=x, n_in=28*28, n_out=10)

	# the cost we minimize during training
	cost = classifier.negative_log_likelihood(y)

	test_model = theano.function(
		inputs  = [index],
		outputs = classifier.errors(y),
		givens={
			x: test_set_x[index * batch_size: (index + 1) * batch_size],
			y: test_set_y[index * batch_size: (index + 1) * batch_size]
		}
	)

	validate_model = theano.function(
		inputs  = [index],
		outputs = classifier.errors(y),
		givens={
			x: valid_set_x[index * batch_size: (index + 1) * batch_size],
			y: valid_set_y[index * batch_size: (index + 1) * batch_size]
		}
	)

	# compute gradient of cost wrt W and b
	g_W = T.grad(cost=cost, wrt=classifier.W)
	g_b = T.grad(cost=cost, wrt=classifier.b)

	# specify how to update the parameters
	updates = [(classifier.W, classifier.W - learning_rate * g_W),
			   (classifier.b, classifier.b - learning_rate * g_b)]

	train_model = theano.function(
		inputs=[index],
		outputs=classifier.negative_log_likelihood(y),
		updates=updates,
		givens={
			x: train_set_x[index * batch_size: (index + 1) * batch_size],
			y: train_set_y[index * batch_size: (index + 1) * batch_size]
		}
	)

	test_losses = [test_model(i) for i in xrange(n_test_batches)]
	test_error = numpy.mean(test_losses) * 100.
	print 'initial test error: %f %%' % test_error

	###############
	# TRAIN MODEL #
	###############
	print '... training the model'

	# early-stopping parameters
	patience              = 5000  # look at this many examples regardless
	patience_increase     = 2     # wait this much longer when a new best is found
	improvement_threshold = 0.995 # a relative improvement of this much is
	                              # considered significant
	validation_frequency  = min(n_train_batches, patience /	2)
								  # go through this many minibatches before checking the
								  # network on the evaluation set; in this case we check
								  # every epoch

	best_validation_loss = numpy.inf
	test_score = 0.
	start_time = time.clock()

	done_looping = False
	epoch = 0
	while (epoch < n_epochs) and (not done_looping):
		epoch = epoch + 1
		for minibatch_index in xrange(n_train_batches):
			minibatch_avg_cost = train_model(minibatch_index)
			iter = (epoch - 1) * n_train_batches + minibatch_index  # iteration number

			if (iter + 1) % validation_frequency == 0:
				# compute zero-one loss on validation set
				validation_losses = [validate_model(i) for i in xrange(n_valid_batches)]
				this_validation_loss = numpy.mean(validation_losses)

				print(
					'epoch %i, minibatch %i/%i, validation error %f %%' %
					(
						epoch,
						minibatch_index + 1,
						n_train_batches,
						this_validation_loss * 100.
					)
				)

				# if we got the best validation score until now
				if this_validation_loss < best_validation_loss:
					# increase patience if loss improvement is good enough
					if this_validation_loss < best_validation_loss * improvement_threshold:
						patience = max(patience, iter * patience_increase)
						print 'new patience: %i' % patience

					best_validation_loss = this_validation_loss

					test_losses = [test_model(i) for i in xrange(n_test_batches)]
					test_score = numpy.mean(test_losses)

					print(
						'     epoch %i, minibatch %i/%i, test error of best model %f %%' %
						(
							epoch,
							minibatch_index + 1,
							n_train_batches,
							test_score * 100
						)
					)

				if patience <= iter:
					done_looping = True
					break

	end_time = time.clock()
	print(
		(
			'Optimization complete with best validation score of %f %%, with test'
			'performance %f %%,'
		) %
		(best_validation_loss * 100., test_score * 100.)
	)
	print 'The code run for %d epochs, with %f epochs/sec' % (
		epoch, 1.* epoch / (end_time - start_time))
	print >> sys.stderr, ('The code for file ' + os.path.split(__file__)[1] +
		' ran for %.1fs' % (end_time-start_time))

if __name__ == '__main__':
	sgd_optimization_mnist()
