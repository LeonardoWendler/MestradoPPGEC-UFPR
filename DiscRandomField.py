# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 22:34:10 2024

@author: Leona
"""

import numpy as np
import pandas as pd
import os 
from numba import jit, njit
from dkle import DiscreteKarhunenLoeveExpansion
from scipy.interpolate import griddata
# from ANSYS_functions import Ortoelastic_prop

# data = pd.read_excel('resincontent.xlsx',header=0)
# stdv = np.std(data.to_numpy(),ddof=1)
# meanval = data.to_numpy().mean()
# RCvec = np.random.normal(0, 1, size=(4))
@jit(nopython=True)
def compute_covariance_matrix(X, s, ell):
    """
    Computes the covariance matrix at ``X``. This computes a
    squared exponential covariance

    :param X:   The evaluation points. It has to be a 2D numpy array of
                dimensions ``num_points x input_dim``.
    :type X:    :class:`numpy.ndarray``
    :param s:   The signal strength of the field. It must be positive.
    :type s:    float
    :param ell: A list of lengthscales. One for each input dimension. The must
                all be positive.
    :type ell:  :class:`numpy.ndarray`
    """
    assert X.ndim == 2
    assert s > 0
    assert ell.ndim == 1
    assert X.shape[1] == ell.shape[0]
    C = np.zeros((X.shape[0], X.shape[0]))

    dx = 0.0
    for i in range(X.shape[0]):
        for j in range(X.shape[0]):
            for k in range(X.shape[1]):
                dx = (X[i, k] - X[j, k])
                C[i, j] += dx * dx
            C[i, j] = np.sqrt(C[i,j])
    return s ** 2 * np.exp(-C/ ell[k])


def sample_RandomField(RCvec, x , y, theta, meanval, stdv):
    """
    Sample the resin content field

    RCvec:  Vector containing the resin content data from standard Gaussian RVs realizations.
    type RCvec:    Numpy ndarray
    
    indexer:  Matrix containing element data [CG X, CG Y, RCID]. (Removed, is directly obtained from excel)
    type indexer:    Numpy ndarray

    x: Vector containing all x coordinates of the discretized domain nodes
    type: Numpy ndarray

    y: Vector containing all y coordinates of the discretized domain nodes
    type: Numpy ndarray
    
    theta: Vector containing correlation length parameters
    
    param meanval:   Resin content data mean.
    type meanval:    float
    param stdv:   Resin content data standard deviation.
    type stdv:    float
    
    Returns an array of floats
    """
    
    #### DOMAIN POINTS

    # Length scales 20 to 100 are recomendations from random field modelling of spatial variability in FRP composite
    # materials paper
    # Srinivas Sriramula
    # School of Engineering, University of Aberdeen, Aberdeen, UK
    # Marios K. Chryssanthopoulos
    ell = np.array([theta, theta])
    # The signal strength of the field
    s = stdv
    # The percentage of energy of the field you want to keep
    # energy = 0.98
    X = np.vstack((x,y)).T
    # Construct the covariance matrix
    C = compute_covariance_matrix(X, s, ell)
    # Compute the eigenvalues and eigenvectors of the field
    eig_values, eig_vectors = np.linalg.eigh(C)
    #sort the eigenvalues and keep only the largest ones
    idx = np.argsort(eig_values)[::-1]
    eig_values = eig_values[idx]
    eig_vectors = eig_vectors[:, idx]
    # The energy of the field up to a particular eigenvalue:
    # energy_up_to = np.cumsum(eig_values) / np.sum(eig_values)
    # The number of eigenvalues giving the desired energy
    # i_max = np.arange(energy_up_to.shape[0])[energy_up_to >= energy][0]
    i_max = RCvec.shape[0]
    # D-KLE of the field and sampling it
    RCvec = RCvec.reshape(RCvec.shape[0],1)
    d_kle = DiscreteKarhunenLoeveExpansion(X, eig_vectors[:,:i_max],
                                               eig_values[:i_max],meanval,RCvec)
    # Gerar uma amostra do campo
    sample = d_kle.sample()[0, :]  # Shape: (N,)
    
    return sample

# dkle_sample = sample_RandomField(RCvec, meanval, stdv)

