# Advanced Machine Learning Project
Code, data, and results for our project in DTU course 02460 in Advanced Machine Learning

So far we are learning to use Theano and trying to replicate the results from the paper [Text Understanding from Scratch](http://arxiv.org/abs/1502.01710) (by X. Zhang, and Y. LeCun). Their own implementation in Torch7 is called [Crepe](https://github.com/zhangxiangxiao/Crepe).

This is really a work in progress.

## Our code

 - **loadtext.py** contains the function `load_text` which can load text from the .csv files that the datasets are provided in. It returns the data as numpy arrays with the text encoded as one-vs-all vectors.
 - **loadtextsparse.py** contains code for loading the text data represented using sparce matrices to save memory.
 - **trainsimple.py** builds and trains a very simple model, that is heavily based on the CNN described in [this deeplearning.net tutorial](http://deeplearning.net/tutorial/lenet.html). Due to memory constraints, this script uses only the training set (70000 samples) for training, validation and test.
 - **simplelasagne.py** is very similar to *trainsimple.py*, but uses ready made layers from [Lasagne](https://github.com/Lasagne/Lasagne) to build the network model. (achieves training error of 40.2%)
 - **simplelasagnesparse.py** trains the model from *simplelasagne.py* on the full dataset, which can be kept in memory due to the sparse representation from *loadtextsparse.py*.
 - files in **tutorials/** are almost identical to the files from the tutorials on DeepLearning.net but without the comments. If you are interested in them, you should look at the versions from [their own repository](https://github.com/lisa-lab/DeepLearningTutorials/tree/master/code) instead. We use the layers they define to build our own simple model in *trainsimple.py*

## Obtaining data

The training scripts assume that the files from the [dbpedia-dataset](http://goo.gl/JyCnZq) (currently the only dataset [without duplicates](http://xzh.me/posts/datasetdup/)) has been unpacked in a directory called `data`, such that the file `test.csv` from the dataset is located at `data/dbpedia_csv/test.csv`.

## Dependencies
These scripts were developed and seem to work with Python
2.7.6 with the following modules:

 - Theano 0.7.0
 - Numpy 1.9.2
 - [Lasagne](https://github.com/Lasagne/Lasagne) (newest version as of 2015-05-15)
