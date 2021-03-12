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
from sklearn.metrics import adjusted_rand_score, mean_squared_error, mean_absolute_error
import argparse

def evaluate_reconstruction(x, encoder, decoder):
    """
    Evaluates the reconstruction error of the given encoder and decoder
    returns the MSE(r, e) where r = encoder(decoder(x)).

    Args:
    x: input to the encoder
    encoder: a function that encodes (x)
    decoder: a function that decodes the output from encoder(x)

    Returns:
    MSE of reconstruction
    """

    recons = decoder(encoder(x))
    recon_mse = mean_squared_error(x, recons)
    return recon_mse

def evaluate_distance(x, encoder, distance):
    """
    Evaluates the distance approximation using the encoder. Approximation is
    based on L2 norm of two codes output from encoder.

    Args:
    x: input to the encoder. Assumes x is a 2d array of shape (m, n) where m =
    number of timeseries, and n = length of each timeseries

    encoder: a function that encodes (x). assumes encoder returns a 2d array of
    shape (m, n) where m = number of timeseries, and n = length of the code

    distance: a function calculates the pair-wise distance of two collections of
    timeseries. t = distance(s1, s2). s1 and s2 are 2d arrays of the same shape
    (m, n). t is a 1d array of length m where t[i] is the distance between
    s1[i] and s2[i].

    Returns: (mae, mse), a tuple where mae = mean absolute error and mse = mean
    squared error.
    """

    assert len(x.shape) == 2

    l = len(x)
    n_pairs =  int( l * math.log2(l))
    idx1 = np.random.randint(0, l, (n_pairs))
    idx2 = np.random.randint(0, l, (n_pairs))
    s1 = x[idx1]
    s2 = x[idx2]

    true_dist = distance(s1, s2)

    codes = encoder(x)
    assert len(codes.shape) == 2

    c1 = codes[idx1]
    c2 = codes[idx2]
    apprx_dist = np.linalg.norm(c1-c2, axis=1)
    dist_mae = mean_absolute_error(apprx_dist, true_dist)
    dist_mse = mean_squared_error(apprx_dist, true_dist)
    return dist_mae, dist_mse

def evaluate_common_nn(x_train, x_test, encoder, distance, nn=10):
    """
    Compares the common nearest neighbors between the baseline distance, and the
    encoder's approximate distance. Approximation is based on L2 norm of two
    codes output from encoder. sklearn NearestNeighbors used to calculate NN.
    returns the average number of common nn between the baseline distance and
    the approximation.

    Args:
    x_train: train data used to look-up nearest neighbors. Assumes a 2d array of
    shape (m, n) where m = number of timeseries, and n = length of each
    timeseries

    x_test: test data whos neighbors to lookup. Assumes a 2d array of shape (m,
    n) where m = number of timeseries, and n = length of each timeseries

    encoder: a function that encodes x_train and x_test. assumes encoder returns
    a 2d array of shape (m, n) where m = number of timeseries, and n = length of
    the code

    distance: distance(x, y) -> float

    nn: the number of nearest neighbors to lookup, default = 10

    Returns: mean number of common nn between baseline distance and the
    approximation
    """

    assert len(x_train.shape) == 2

    # True nearest neighbors
    neighbors = NearestNeighbors(n_neighbors=nn, metric=distance)
    neighbors.fit(x_train)
    true = neighbors.kneighbors(x_test, return_distance=False)

    # Approximations
    code_train = encoder(x_train)
    code_test = encoder(x_test)

    neighbors = NearestNeighbors(n_neighbors=nn)
    neighbors.fit(code_train)
    apprx = neighbors.kneighbors(code_test, return_distance=False)

    intersect = lambda x, y: len(set(x).intersection(set(y)))
    nn_common = np.array([intersect(x, y) for x, y in zip(true, apprx) ])

    return nn_common.mean()

def evaluate_clustering_ri(x_train, x_test, encoder, baseline, n_clusters):
    """
    Evaluates the clustering of the encoder's output vs. a baseline clustering
    strategy. Clustering of encoder's output uses KMeans and the L2 norm of the
    output.

    Args:
    x_train: train data used to look-up nearest neighbors. Assumes a 2d array of
    shape (m, n) where m = number of timeseries, and n = length of each
    timeseries

    x_test: test data whos neighbors to lookup. Assumes a 2d array of shape (m,
    n) where m = number of timeseries, and n = length of each timeseries

    encoder: a function that encodes x_train and x_test. assumes encoder returns
    a 2d array of shape (m, n) where m = number of timeseries, and n = length of
    the code

    baseline: a function that clusters x_train and x_test. will take x_train and
    x_test as inputs. should cluster inputs to the n_clusters arguments. returns
    a 1D array of the clusters assigned to each item in the input

    n_clusters: the number of clusters

    Returns: adjusted rand index score of the two clustering strategies
    """

    assert len(x_train.shape) == 2

    baseline_clusters = baseline(x_test)

    code_train = encoder(x_train)
    code_test = encoder(x_test)
    kmeans = KMeans(n_clusters=n_clusters).fit(code_train)
    approximate_clusters = kmeans.predict(code_test)

    return adjusted_rand_score(baseline_clusters, approximate_clusters)
