# Advanced Machine Learning Project
Code for our project in DTU course 02460 in Advanced Machine Learning.

By [Linards Kalnins](https://github.com/linardslinardslinards) and [Jonas Meinertz Hansen](https://github.com/Styrke)

## Contents of the repository

Machinery:
 - **loadtext.py** contains the function `load_text` which can load text from the .csv files that the datasets are provided in. It returns the text as a sparse numpy arrays with the text encoded as one-vs-all vectors and the labels as a regular numpy array.
 - **smtpmail** Has a function, `send_email`, which can send emails through the SMTP protocol. This functionality is used by traintool.py to send emails with updates on training, which can be handled by services like IFTTT to do all kinds of awesome things like updating a file in your Dropbox with the newest results.
 - **traintool.py** Contains a function, `train_model`, that takes care of the actual training of the model (which is mostly the same for all of them anyways.) It can change the learning rate on a schedule, send emails with updates on the training, and much more.

Models for training:
 - **train_demo.py** is for testing purposes, and is a good first model to try to see that everything works. It only uses the smaller test set for everything, so it loads much faster than the other models. It is also constrained to 5 epochs.
 - **train_convolutional.py** is a network with 2 convolutional layers and 2 fully connected layers.
 - **train_convolutional_v2.py** is mostly the same as *train_convolutional.py*, but it utilizes rectified activations and has a dropout layer.
 - **train_full.py** should be the equivalent of the models from [1], but unfortunately we aren't able to run it due to having too little memory on the GPU we use.
 - **train_LSTM.py** is for training a simple recurrent model with LSTM cells.

## Notes for running scripts
This code is made to be executed on cloud servers, and include a script to send emails with updates on training progress using the SMTP protocol. Before running the script you should either rename `mailconfig.py.blank` to `mailconfig.py` and fill in your own details or comment out all lines in `traintool.py` that use the function `send_email`.

Before training a model make sure to make a folder called `parameters` to save the parameters to after training. To train e.g. the demo model simply run the script with: `python train_demo.py`

## Obtaining data

The training scripts assume that the files from the [dbpedia-dataset](http://goo.gl/JyCnZq) (currently the only dataset [without duplicates](http://xzh.me/posts/datasetdup/)) has been unpacked in a directory called `data`, such that the file `test.csv` from the dataset is located at `data/dbpedia_csv/test.csv`.

## Dependencies
These scripts were developed and seem to work with Python
2.7.6 with the following modules:

 - Theano 0.7.0
 - Numpy 1.9.2
 - A recent version of [Lasagne](https://github.com/Lasagne/Lasagne). (In order to train the LSTM model, you'll need a version that has support for recurrent layers. We have been using [this fork](https://github.com/craffel/nntools/tree/recurrent) for development.)
