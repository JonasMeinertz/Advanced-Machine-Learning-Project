import csv
import numpy
import scipy.sparse
import theano

def load_text(file='train.csv', samples=560000, length=130):
    permutation = numpy.random.permutation(samples)
    alphabet = list("abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\|_@#$%^&*~`+-=<>()[]{}")

    # initialize arrays for building sparce matrices
    values = numpy.zeros(samples*length, dtype=theano.config.floatX)
    rows = numpy.zeros(samples*length, dtype=numpy.int_)
    cols = numpy.zeros(samples*length, dtype=numpy.int_)

    reader = csv.reader(open('data/dbpedia_csv/'+file, 'rb'))
    y = numpy.zeros([samples], dtype=numpy.int_)
    arrit = 0 # for keeping track of the number of non-zero entries
    for it in permutation:
        row = reader.next()
        y[it] = int(row[0])-1       # the label
        string = row[1] + row[2]    # title + content
        string = string.lower()

        # convert characters to one-hot vectors
        for idx, char in enumerate(string[0:length]):
            if char in alphabet:
                values[arrit] = 1
                rows[arrit] = it
                cols[arrit] = idx + alphabet.index(char)*length
                arrit += 1

    # create the sparse matrix
    x = scipy.sparse.csr_matrix((values[0:arrit],
                                 (rows[0:arrit], cols[0:arrit])),
                                shape=(samples, length*69))

    return (x, y)
