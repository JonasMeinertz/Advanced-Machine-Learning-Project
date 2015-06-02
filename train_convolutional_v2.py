import lasagne
from traintool import train_model
from loadtext import load_text

# same as convolutional but with rectify nonlinearities, dropout layer, and
# learning rate schedule.

NAME = 'convolutional_v2'
BATCH_SIZE = 128

LEARNING_RATE_SCHEDULE = {
    0: 0.01,
    3: 0.005,
    6: 0.0025,
    9: 0.00125
}

# loading text data
print '... loading text data'
train_set = load_text(file='train.csv', samples=560000, length=130)
test_set = load_text(file='test.csv', samples=70000, length=130)
data = (
    (train_set[0][:490000], train_set[1][:490000]),
    (train_set[0][490000:], train_set[1][490000:]),
    test_set
)

# define model
layer0_flat = lasagne.layers.InputLayer(shape=(BATCH_SIZE, 69 * 130))
layer0_input = lasagne.layers.ReshapeLayer(layer0_flat, (BATCH_SIZE, 1, 130, 69))
layer0 = lasagne.layers.Conv2DLayer(
    layer0_input,
    num_filters  = 50,
    filter_size  = (5, 1),
    nonlinearity = lasagne.nonlinearities.rectify
)
layer0_pool = lasagne.layers.MaxPool2DLayer(layer0, pool_size=(3, 1))
layer1 = lasagne.layers.Conv2DLayer(
    layer0_pool,
    num_filters  = 50,
    filter_size  = (3, 1),
    nonlinearity = lasagne.nonlinearities.rectify
)
layer1_pool = lasagne.layers.MaxPool2DLayer(layer1, pool_size=(4, 1))
layer2 = lasagne.layers.DenseLayer(
    layer1_pool,
    num_units = 500,
    nonlinearity = lasagne.nonlinearities.rectify
)
dropout = lasagne.layers.DropoutLayer(layer2)
layer3 = lasagne.layers.DenseLayer(
    dropout,
    num_units = 14,
    nonlinearity = lasagne.nonlinearities.softmax
)

train_model(
    layer3,
    data,
    name=NAME,
    learning_rate_schedule=LEARNING_RATE_SCHEDULE,
    batch_size=BATCH_SIZE
)
