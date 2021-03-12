import sys, os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import py_ts_data

import tensorflow as tf
from tensorflow.signal import fft, ifft
from tensorflow.math import conj
from tensorflow import norm

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from sklearn.cluster import KMeans
import argparse
import evaluation_utils as ev

PARSER = argparse.ArgumentParser()
PARSER.add_argument('-d', '--dataset', default=None, required=True, help="dataset to run")
PARSER.add_argument('-m', '--model', default="sample_model", required=False, help="model to run")
ARGS = PARSER.parse_args()

DATA = ARGS.dataset
MODELS_PATH = ARGS.model

ENCODER = tf.keras.models.load_model(os.path.join(MODELS_PATH, DATA, "encoder"))
DECODER = tf.keras.models.load_model(os.path.join(MODELS_PATH, DATA, "decoder"))
X_TRAIN, Y_TRAIN, X_TEST, Y_TEST, _ = py_ts_data.load_data(DATA, variables_as_channels=True)
# all are read in with 3 dims, last is num of variables in the TS
assert len(X_TRAIN.shape) == 3
# we care only about univariate TS
assert X_TRAIN.shape[2] == 1
X_TRAIN = np.squeeze(X_TRAIN, axis=2)
X_TEST = np.squeeze(X_TEST, axis=2)

N_NEIGHBORS = 10
N_CLUSTERS  = len(set(Y_TRAIN))
CLUSTERING = KMeans(N_CLUSTERS).fit(X_TRAIN)

def encoder(x):
    assert len(x.shape) == 2
    x = x[..., np.newaxis]
    codes = ENCODER(x)
    return codes.numpy()

def decoder(x):
    assert len(x.shape) == 2
    result = DECODER(x).numpy()
    assert len(result.shape) == 3
    assert result.shape[2] == 1
    result = np.squeeze(result, axis=2)
    return result

def distance_collection(x, y):
    assert len(x.shape) == 2
    assert len(y.shape) == 2
    assert len(x) == len(y)
    return np.linalg.norm(x-y, axis=1)

def distance_timeseries(x, y):
    assert len(x.shape) == 1
    assert len(y.shape) == 1
    assert len(x) == len(y)
    return np.linalg.norm(x-y)

def clustering(x):
    assert len(x.shape) == 2
    return CLUSTERING.predict(x)

if __name__ == "__main__":
    recon = ev.evaluate_reconstruction(X_TEST, encoder, decoder)
    dist = ev.evaluate_distance(X_TEST, encoder, distance_collection)
    common = ev.evaluate_common_nn(X_TRAIN, X_TEST, encoder, distance_timeseries, N_NEIGHBORS)
    ri = ev.evaluate_clustering_ri(X_TRAIN, X_TEST, encoder, clustering, N_CLUSTERS)
    print("{}, reconstruction: {:.3f}, distance mse: {:.3f}, distance mae: {:.3f}, common nn: {:.3f}, rand index: {:.3f}".format(DATA, recon, dist[0], dist[1], common, ri))



