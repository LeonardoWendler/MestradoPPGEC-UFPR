# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 00:15:26 2024

@author: User
"""
import scipy as sc
from scipy.stats import qmc
from scipy import stats
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from DiscRandomField import sample_RandomField, compute_covariance_matrix
import joblib
import time
from PlotSMCfunc import PlotSMC
import h5py

" Reimportando o modelo de GP ja treinado "
x_points = pd.read_excel('elem_data_fileatt.xlsx',header=0,sheet_name="RCIndTrue").to_numpy()[:,0]
y_points = pd.read_excel('elem_data_fileatt.xlsx',header=0,sheet_name="RCIndTrue").to_numpy()[:,1]
# Objective function

def obj_func(theta, eval_gradient=True):
    # theta: current kernel hyperparameters
    # Computes: -log marginal likelihood under current kernel
    lml, grad = gpunif.log_marginal_likelihood(
        theta, eval_gradient=True, clone_kernel=False
    )
    return -lml, -grad

def custom_optimizer(obj_func, initial_theta, bounds):
    opt_res = sc.optimize.minimize(
        obj_func,
        initial_theta,
        bounds=bounds,
        method="L-BFGS-B",
        jac=True,
        options={'maxiter': 500})
    theta_opt, func_min = opt_res.x, opt_res.fun
    return theta_opt, func_min

" METAMODELO DO MODELO MEF QUE CONSIDERA HOMOGENEIDADE "
# load do modelo que considera a heterogeneidade
gpunif = joblib.load("gpruniformmodel.pkl")

" SMC COM AMOSTRAGEM LHS "
means = np.array([2000., 2000., 2000., 2250.,2250.,2250.,2500.,2500.,2500.,3000.,
                  3000.,3000.,3300.,3300.,3300.,3900.,3900.,3900.])

seeds = np.array([900989, 803680, 211104, 737988, 164545, 854678, 57199, 431163,
                  299802, 729438, 649660, 220776, 785067, 404554, 399329, 
                  176086, 945493, 409787])

" SMC GP MODELO HOMOGENEO "
pf = np.zeros(means.shape[0])
COV = np.zeros(means.shape[0])
FIndunif = []
for i in range(0, means.shape[0]):
    seed2 = seeds[i]
    mean = means[i]
    np.random.seed(seed2)
    sampler2 = qmc.LatinHypercube(2,seed=seed2)
    sample2 = sampler2.random(n=150000)
    ns = sample2.shape[0]
    sample2[:,0] = sc.stats.norm.ppf(sample2[:,0], loc=mean, scale=0.1*mean)
    sample2[:,1] = sc.stats.norm.ppf(sample2[:,1], loc=26.762962962962973, scale=4.289119903403516)
    responses_unif = gpunif.predict(sample2[:,1].reshape(-1,1), return_std=False)
    FIndunif.append(responses_unif*0.25 - sample2[:,0] <= 0)
    pf[i] = np.mean(FIndunif[i])
    COV[i] = np.sqrt(pf[i] * (1 - pf[i]) / ns)/pf[i]*100

print("pf: ", np.mean(FIndunif))
print("COV: ", COV)
print("Media RC: ", np.mean(sample2[:,1]))
