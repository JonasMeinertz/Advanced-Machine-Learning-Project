import csv
import numpy

def load_text(length=1014, rng=numpy.random):
    alphabet = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\|_@#$%^&*~`+-=<>()[]{}"
    alphabet = list(alphabet)
    permutation = rng.permutation(70000)

    reader = csv.reader(open('data/dbpedia_csv/test.csv', 'rb'))
    x = numpy.zeros([70000, length * 69], dtype=numpy.bool_)
    y = numpy.zeros([70000], dtype=numpy.int_)

    for it in permutation:
        row = reader.next()
        y[it] = int(row[0])-1
        #y[it][int(row[0])-1] = 1
        string = row[1] + row[2]
        string = string.lower()

        for idx, char in enumerate(string[0:length]):
            if char in alphabet:
                x[it][idx*69 + alphabet.index(char)] = 1

    xs = numpy.split(x, [50000,60000])
    ys = numpy.split(y, [50000,60000])
    return zip(xs, ys) #(xs[0], ys[0]), (xs[1], ys[1]), (xs[2], ys[2])

if __name__ == "__main__":
    a = load_text(200)
