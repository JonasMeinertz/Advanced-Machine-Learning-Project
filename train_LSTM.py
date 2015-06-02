import lasagne
from traintool import train_model
from loadtext import load_text

NAME = 'recurrent'
BATCH_SIZE = 128

LEARNING_RATE_SCHEDULE = {
    0: 0.005,
    3: 0.0025,
    6: 0.00125
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
layer_flat = lasagne.layers.InputLayer(shape=(BATCH_SIZE, 69 * 130))
layer_input = lasagne.layers.ReshapeLayer(layer_flat, (BATCH_SIZE, 130, 69))

layer_lstm = lasagne.layers.LSTMLayer(
    layer_input,
    30,
    learn_init=True,
    gradient_steps = 10
)

layer_out = lasagne.layers.DenseLayer(
    layer_lstm,
    num_units = 14,
    nonlinearity = lasagne.nonlinearities.softmax
)

train_model(
    layer_out,
    data,
    name=NAME,
    learning_rate_schedule=LEARNING_RATE_SCHEDULE,
    batch_size=BATCH_SIZE
)
