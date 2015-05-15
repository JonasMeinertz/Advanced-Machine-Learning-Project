import csv
import numpy
import scipy.sparse
import theano
import time

def load_text(data='data/dbpedia_csv/train.csv', samples=560000, length=1014,
              rng=numpy.random):
    start_time = time.clock()
    alphabet = list(
        "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\|_@#$%^&*~`+-=<>()[]{}"
        )
    permutation = rng.permutation(samples)

    reader = csv.reader(open(data, 'rb'))
    y = numpy.zeros([samples], dtype=numpy.int_)

    values = numpy.zeros(samples*length, dtype=theano.config.floatX)
    rows = numpy.zeros(samples*length, dtype=numpy.int_)
    cols = numpy.zeros(samples*length, dtype=numpy.int_)
    arrit = 0
    for it in permutation:
        row = reader.next()
        y[it] = int(row[0])-1
        #y[it][int(row[0])-1] = 1
        string = row[1] + row[2]
        string = string.lower()

        for idx, char in enumerate(string[0:length]):
            if char in alphabet:
                values[arrit] = 1
                rows[arrit] = it
                cols[arrit] = idx*69 + alphabet.index(char)
                arrit += 1

    x = scipy.sparse.csr_matrix((values[0:arrit],
                                 (rows[0:arrit], cols[0:arrit])),
                                shape=(samples, length*69))

    #print (x.data.nbytes + x.indices.nbytes + x.indptr.nbytes)/1000

    #print '    in %.1f seconds' % (time.clock() - start_time)

    return (x, y) #(xs[0], ys[0]), (xs[1], ys[1]), (xs[2], ys[2])

if __name__ == "__main__":
    (x, y) = load_text(samples=10)
    print y
