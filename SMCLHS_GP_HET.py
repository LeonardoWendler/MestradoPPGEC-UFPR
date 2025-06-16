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
    lml, grad = gphet.log_marginal_likelihood(
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

" METAMODELO DO MODELO MEF QUE CONSIDERA HETEROGENEIDADE "
# load do modelo que considera a heterogeneidade
gphet = joblib.load("gprmodelv2.pkl")

def obj_func(theta, eval_gradient=True):
    # theta: current kernel hyperparameters
    # Computes: -log marginal likelihood under current kernel
    lml, grad = gpunif.log_marginal_likelihood(
        theta, eval_gradient=True, clone_kernel=False)
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
#Para caso 1, theta  = 20
seed = 290251
# seed = 278787
# #Para caso 1, theta  = 90
# seed = 740540
# seed = 857858
# #Para caso 1, theta  = 1000
# seed = 977710
# seed = 621705

mean = 2000.

# #Para caso 2, theta  = 20
# seed = 657667
# seed = 815202
# #Para caso 2, theta  = 90
# seed = 664608
# seed = 498614
# #Para caso 2, theta  = 1000
# seed = 835037
# seed = 600232

# mean = 2500.

# #Para caso 3, theta  = 20
# seed = 649636
# seed = 889113
# #Para caso 3, theta  = 90
# seed = 142492
# seed = 559493
# #Para caso 3, theta  = 1000
# seed = 49524
# seed = 605242

# mean = 3900.

means = np.array([2000., 2000., 2000., 2000., 2000., 2000.,
                   2500.,2500.,2500.,2500.,2500.,2500.,
                   3900.,3900.,3900.,3900.,3900.,3900. ])

seeds = np.array([290251, 740540,857858, 977710,621705, 657667, 815202,
                  664608, 498614, 835037, 600232, 649636, 889113,
                  142492, 559493, 49524, 605242 ])

caso = np.array([1,1, 2,2, 3,3, 4,4, 5,5 ,6,6, 7,7, 8,8, 9,9])

thetas = np.array([20.,20., 20., 20., 20., 20., 90., 90.,90., 90.,90., 90., 1000., 1000., 
                  1000., 1000., 1000., 1000.])

for j in range(0, caso.shape[0]):
    
    np.random.seed(seed=seeds[j])
    sampler = qmc.LatinHypercube(24,seed=seeds[j])
    sample = sampler.random(n=int(15e5))
    sample[:,0] = sc.stats.norm.ppf(sample[:,0], loc=means[j], scale=0.1*means[j])
    sample[:,1:] = sc.stats.norm.ppf(sample[:,1:], loc=0, scale=1)
    " SMC GP MODELO HETEROGENEO "
    " Dados para campo aleatorio "

    responses = np.zeros((sample.shape[0]))
    timecount = 0
    ns = sample.shape[0]
    Rcvec = np.zeros((ns,24))
    responses = np.zeros((ns))
    FInd = np.zeros((ns))
    
    for i in range(0,sample.shape[0]):
        Rcvec[i,:] = sample_RandomField(sample[i,1:], x_points, y_points, thetas[j], 26.762962962962973, 4.289119903403516)
    
    with h5py.File("sample"+str(caso[j])+"seed"+str(seeds[j])+"theta"+str(thetas[j])+".h5", "w") as f:
        f.create_dataset("sample"+str(caso[j])+"seed"+str(seeds[j])+"theta"+str(thetas[j]), data=Rcvec)
    f.close()
    
    print("Status: ",(i+1),"/", caso.shape[0])

hf = h5py.File("sample"+str(caso)+"seed"+str(see)+"theta"+str(theta)+".h5", 'r')
parts = 50
chunk_size = Rcvec.shape[0] // parts  # 1/50 do total
timecount = 0
for i in range(50):
    start = i * chunk_size
    end = (i + 1) * chunk_size if i < int(parts-1) else Rcvec.shape[0]  # Garante o restante na última parte   
    ini=time.time()
    responses[start:end] = gphet.predict(hf["sample"+str(caso)+"seed"+str(see)+"theta"+str(theta)][start:end], return_std=False)
    timecount+=time.time()-ini
    print("Status: ",(i+1),"/", parts, "Elapsed time: ",round(timecount,0)," s", "\n Remaining time: ", round((timecount/(i+1))*(parts-i-1),0)," s")

# Para amostras pequenas a medias:
FInd = (responses*0.25 - sample[:,0] <= 0)
pf_het = np.mean(FInd)

print("pf: ", np.mean(FInd))
print("Media RC: ", np.mean(Rcvec))
print("Media RC: ", np.mean(Rcvec[:,0]))

plotHeterog = PlotSMC(np.arange(80000,ns+100,step=100),FInd, 0.05)

COV_arr = plotHeterog.COV_arr()
print(COV_arr[-1]*100)
COV_het = COV_arr[-1]*100
STD_arr = plotHeterog.std_arr()
VAR_arr = plotHeterog.VAR_arr()
pf_est = plotHeterog.pf_est()
print(pf_est[-1])

plotHeterog.plot_pf()
plotHeterog.plot_pf_log10()
plotHeterog.plot_COVpf()