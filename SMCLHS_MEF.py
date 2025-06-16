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
from DiscRandomField import sample_RandomField
from ANSYS_functions import evaluate_LHS, evaluate_uniform
import openpyxl as xl
import time
from PlotSMCfunc import PlotSMC
import joblib
import multiprocessing as mp

"Dados para campo aleatorio"
x_points = pd.read_excel('elem_data_fileatt.xlsx',header=0,sheet_name="RCIndTrue").to_numpy()[:,0]
y_points = pd.read_excel('elem_data_fileatt.xlsx',header=0,sheet_name="RCIndTrue").to_numpy()[:,1]

seeds = np.array([4266645, 3566911,   29815, 489822,  657667, 835037,
                  664608, 498614, 815202, 461433,  600232, 649636, 889113,
                  142492, 559493, 49524, 605242, 5819737])
means = np.array([3000., 3000., 3000., 3000., 3000.,
                   3000.,3000.,3000.,3000.,
                   3300.,3300.,3300.,3300.,3300., 3300., 3300., 3300., 3300.])

thetas = np.array([20.,20., 20.,  90., 90., 90. , 1000., 1000., 1000., 
                   20.,20., 20.,  90., 90., 90. , 1000., 1000., 1000.])

el = np.array([24,24, 24, 20, 20,20, 6, 6, 6, 
               24,24, 24, 20, 20,20, 6, 6, 6])

FInd = []
responseslist = []
pf = np.zeros(thetas.shape[0])
CoV = np.zeros(thetas.shape[0])
for j in range(9, seeds.shape[0]):
    np.random.seed(seed=seeds[j])
    sampler = qmc.LatinHypercube(el[j],seed=seeds[j])
    sample = sampler.random(n=int(150))
    mean = means[j]
    sample[:,0] = sc.stats.norm.ppf(sample[:,0], loc=mean, scale=0.1*mean)
    sample[:,1:] = stats.norm.ppf(sample[:,1:], loc=0., scale=1.)
    responses = np.zeros((sample.shape[0]))
    timecount = 0
    ns = sample.shape[0]
    Rcvec = np.zeros((sample.shape[0],24))
    
    for i in range(0,sample.shape[0]):
        ti = time.time()
        Rcvec[i,:] = sample_RandomField(sample[i,1:], x_points, y_points, thetas[j], 26.762962962962973, 4.289119903403516)
        responses[i] = evaluate_LHS(Rcvec[i,:])
        # responses[i] = gphet.predict(Rcvec[i,:].reshape(1, -1),return_std = False)[0]
        print("Progresso: ",i+1,"/",sample.shape[0]," (", float(" %2f"%((i+1)/sample.shape[0]*100)),"%)")
        tf = time.time()
        timecount+=tf-ti
        print("Elapsed time: ",round(timecount,ndigits=2),"s", "Mean time per evaluation: ",round(timecount/(i+1),ndigits=2), "s")
        print("Remaining time: ", round((timecount/(i+1))*(ns-i-1),2),"s (", round((timecount/(i+1))*(ns-i-1)/(3600),2), "h)")
        pfit = np.mean(responses[:i]*0.25 - sample[:i,0] <= 0)
        print("pf = ",pfit)
        try:
            CoViter = np.sqrt(pfit * (1 - pfit) / (i+1))/pfit*100
            print("CoV = ",CoViter)
        except:
            pass
    # FInd.append(responses*0.25 - sample[:,0] <= 0)
    # pf[j] = np.mean(FInd[j])
    # CoV[j] = np.sqrt(pf[j] * (1 - pf[j]) / ns)/pf[j]*100
    print(str(i)+"/"+str(sample.shape[0]))
    # print("pf = ",np.mean(FInd[j]))
    # print("CoV = ",CoV[j])
    responseslist.append(responses)


seedt =  461433
np.random.seed(seed=seedt)
sampler = qmc.LatinHypercube(24,seed=seedt)
sample = sampler.random(n=int(150))
mean = 3300
sample[:,0] = sc.stats.norm.ppf(sample[:,0], loc=mean, scale=0.1*mean)
sample[:,1:] = stats.norm.ppf(sample[:,1:], loc=0., scale=1.)
i=0
Rcvectest = sample_RandomField(sample[i,1:], x_points, y_points, 20., 26.762962962962973, 4.289119903403516)
responsestest = evaluate_LHS(Rcvectest)

responses = np.zeros((sample.shape[0]))
timecount = 0
ns = sample.shape[0]
Rcvec = np.zeros((sample.shape[0],24))
j = 0
for i in range(0,sample.shape[0]):
    ti = time.time()
    Rcvec[i,:] = sample_RandomField(sample[i,1:], x_points, y_points, thetas[j], 26.762962962962973, 4.289119903403516)
    responses[i] = evaluate_LHS(Rcvec[i,:])
    # responses[i] = gphet.predict(Rcvec[i,:].reshape(1, -1),return_std = False)[0]
    print("Progresso: ",i+1,"/",sample.shape[0]," (", float(" %2f"%((i+1)/sample.shape[0]*100)),"%)")
    tf = time.time()
    timecount+=tf-ti
    print("Elapsed time: ",round(timecount,ndigits=2),"s", "Mean time per evaluation: ",round(timecount/(i+1),ndigits=2), "s")
    print("Remaining time: ", round((timecount/(i+1))*(ns-i-1),2),"s (", round((timecount/(i+1))*(ns-i-1)/(3600),2), "h)")
    pfit = np.mean(responses[:i]*0.25 - sample[:i,0] <= 0)
    print("pf = ",pfit)
    try:
        CoViter = np.sqrt(pfit * (1 - pfit) / (i+1))/pfit*100
        print("CoV = ",CoViter)
    except:
        pass        