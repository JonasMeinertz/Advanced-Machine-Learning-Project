import lasagne
from traintool import train_model
from loadtext import load_text

NAME = 'full'
BATCH_SIZE = 128
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
nkerns = 256

# loading text data
print '... loading text data'
# train_set = load_text(file='train.csv', samples=560000, length=1014)
# test_set = load_text(file='test.csv', samples=70000, length=1014)
# data = (
#     (train_set[0][:490000], train_set[1][:490000]),
#     (train_set[0][490000:], train_set[1][490000:]),
#     test_set
# )
test_set = load_text(file='test.csv', samples=70000, length=130)
data = (
    (test_set[0][:50000], test_set[1][:50000]),
    (test_set[0][50000:60000], test_set[1][50000:60000]),
    (test_set[0][60000:], test_set[1][60000:])
)

# define model
layer0_flat = lasagne.layers.InputLayer(shape=(BATCH_SIZE, 69 * 1014))
layer0_input = lasagne.layers.ReshapeLayer(layer0_flat, (BATCH_SIZE, 1, 1014, 69))

layer1 = lasagne.layers.Conv2DLayer(
    layer0_input,
    num_filters  = nkerns,
    filter_size  = (7, 1),
    nonlinearity = lasagne.nonlinearities.rectify,
    W            = lasagne.init.Normal(0.05, 0)
)
layer1_pool = lasagne.layers.MaxPool2DLayer(layer1, pool_size=(3, 1))

layer2 = lasagne.layers.Conv2DLayer(
    layer1_pool,
    num_filters  = nkerns,
    filter_size  = (7, 1),
    nonlinearity = lasagne.nonlinearities.rectify,
    W            = lasagne.init.Normal(0.05, 0)
)
layer2_pool = lasagne.layers.MaxPool2DLayer(layer2, pool_size=(3, 1))

layer3 = lasagne.layers.Conv2DLayer(
    layer2_pool,
    num_filters  = nkerns,
    filter_size  = (3, 1),
    nonlinearity = lasagne.nonlinearities.rectify,
    W            = lasagne.init.Normal(0.05, 0)
)

layer4 = lasagne.layers.Conv2DLayer(
    layer3,
    num_filters  = nkerns,
    filter_size  = (3, 1),
    nonlinearity = lasagne.nonlinearities.rectify,
    W            = lasagne.init.Normal(0.05, 0)
)

layer5 = lasagne.layers.Conv2DLayer(
    layer4,
    num_filters  = nkerns,
    filter_size  = (3, 1),
    nonlinearity = lasagne.nonlinearities.rectify,
    W            = lasagne.init.Normal(0.05, 0)
)

layer6 = lasagne.layers.Conv2DLayer(
    layer5,
    num_filters  = nkerns,
    filter_size  = (3, 1),
    nonlinearity = lasagne.nonlinearities.rectify,
    W            = lasagne.init.Normal(0.05, 0)
)
layer6_pool = lasagne.layers.MaxPool2DLayer(layer6, pool_size=(3, 1))

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

# train it
train_model(
    layer9,
    data,
    name=NAME,
    learning_rate_schedule=LEARNING_RATE_SCHEDULE,
    batch_size=BATCH_SIZE
)
